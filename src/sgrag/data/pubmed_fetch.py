"""NCBI Entrez efetch client and PubMed XML parser.

Pilot corpus acquisition only: fetches PubMed abstracts for an explicit,
deterministic PMID list (see src/sgrag/data/pilot.py). No query expansion,
no silent substitution — what you list is what you get.
"""

import time
import xml.etree.ElementTree as ET

import requests

EUTILS_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
DEFAULT_TOOL = "sgrag_pilot"
DEFAULT_BATCH_SIZE = 50
# NCBI allows 3 requests/second without an API key; stay well under it.
DEFAULT_DELAY_S = 0.4


def _single_attempt(batch, delay_s, timeout, tool, email):
    """One efetch request with a FRESH session.

    A fresh session (plus Connection: close) is deliberate: some egress
    proxies mishandle HTTP keep-alive reuse, where the first request on a
    session succeeds and subsequent reused connections hang until read
    timeout. Fresh sessions make every attempt independent.
    """
    params = {
        "db": "pubmed",
        "id": ",".join(batch),
        "retmode": "xml",
        "rettype": "abstract",
        "tool": tool,
    }
    if email:
        params["email"] = email
    sess = requests.Session()
    try:
        resp = sess.get(EUTILS_EFETCH, params=params, timeout=timeout,
                        headers={"Connection": "close"})
        return resp.status_code, resp.content if resp.status_code == 200 else None, None
    except requests.RequestException as exc:  # timeout / connection drop
        return None, None, f"{type(exc).__name__}: {exc}"
    finally:
        sess.close()


def fetch_batches(pmids, batch_size=DEFAULT_BATCH_SIZE, delay_s=DEFAULT_DELAY_S,
                  timeout=30, tool=DEFAULT_TOOL, email=None, max_retries=3):
    """Fetch PubMed records from Entrez in batches.

    Args:
        pmids: ordered list of PMID strings to fetch.
        batch_size: PMIDs per efetch request.
        delay_s: pause between requests (rate-limit politeness).
        email: contact email sent to NCBI (optional; from SGRAG_CONTACT_EMAIL).

    Returns:
        List of dicts, one per batch, in input order:
        {"pmids": [...], "xml": bytes|None, "http_status": int|None,
         "attempts": int, "error": str|None}
    """
    results = []
    batches = [pmids[i:i + batch_size] for i in range(0, len(pmids), batch_size)]
    for bi, batch in enumerate(batches):
        attempts, error, xml_bytes, status = 0, None, None, None
        for attempt in range(1, max_retries + 1):
            attempts = attempt
            status, xml_bytes, req_error = _single_attempt(
                batch, delay_s, timeout, tool, email)
            if req_error is None and status == 200:
                error = None
                break
            error = req_error or f"HTTP {status}"
            if attempt < max_retries:
                time.sleep(delay_s * (2 ** attempt))
        results.append({
            "pmids": list(batch),
            "xml": xml_bytes,
            "http_status": status,
            "attempts": attempts,
            "error": error,
        })
        if bi < len(batches) - 1:
            time.sleep(delay_s)
    return results


def _text(element):
    if element is None or element.text is None:
        return ""
    return " ".join(element.text.split())


def parse_pubmed_xml(xml_bytes):
    """Parse one efetch XML payload into article records.

    Returns a list of dicts:
    {"pmid": str|None, "title": str|None, "abstract": str|None,
     "journal": str|None, "year": int|None, "mesh_terms": [str],
     "parse_ok": bool}
    Records without an abstract keep abstract=None (flagged, never dropped).
    """
    records = []
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return records
    for article in root.findall("PubmedArticle"):
        rec = {"pmid": None, "title": None, "abstract": None,
               "journal": None, "year": None, "mesh_terms": [],
               "parse_ok": True}
        try:
            citation = article.find("MedlineCitation")
            if citation is None:
                rec["parse_ok"] = False
                records.append(rec)
                continue
            pmid_el = citation.find("PMID")
            rec["pmid"] = _text(pmid_el) or None

            art = citation.find("Article")
            if art is not None:
                title = _text(art.find("ArticleTitle")) or None
                rec["title"] = title
                abs_texts = []
                for ab in art.findall("Abstract/AbstractText"):
                    label = ab.get("Label")
                    txt = _text(ab)
                    if txt:
                        abs_texts.append(f"{label}: {txt}" if label else txt)
                rec["abstract"] = "\n".join(abs_texts) if abs_texts else None
                journal = art.find("Journal")
                if journal is not None:
                    rec["journal"] = _text(journal.find("Title")) or None
                    pubdate = journal.find("JournalIssue/PubDate")
                    if pubdate is not None:
                        year_txt = _text(pubdate.find("Year"))
                        if year_txt.isdigit():
                            rec["year"] = int(year_txt)

            for mh in citation.findall("MeshHeadingList/MeshHeading"):
                desc = mh.find("DescriptorName")
                term = _text(desc)
                if term:
                    rec["mesh_terms"].append(term)
        except Exception:  # never let one bad article kill the batch
            rec["parse_ok"] = False
        records.append(rec)
    return records
