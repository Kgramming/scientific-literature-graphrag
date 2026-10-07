"""Unit tests: PubMed XML parsing (fixture, no network)."""
from sgrag.data.pubmed_fetch import parse_pubmed_xml

FIXTURE = b"""<?xml version="1.0"?>
<PubmedArticleSet>
<PubmedArticle>
 <MedlineCitation><PMID>111</PMID>
  <Article>
   <ArticleTitle>First title</ArticleTitle>
   <Abstract><AbstractText Label="BACKGROUND">Bg text.</AbstractText><AbstractText Label="RESULTS">Res text.</AbstractText></Abstract>
   <Journal><Title>J One</Title><JournalIssue><PubDate><Year>2020</Year></PubDate></JournalIssue></Journal>
  </Article>
  <MeshHeadingList><MeshHeading><DescriptorName>Diabetes</DescriptorName></MeshHeading></MeshHeadingList>
 </MedlineCitation>
</PubmedArticle>
<PubmedArticle>
 <MedlineCitation><PMID>222</PMID>
  <Article>
   <ArticleTitle>Second title</ArticleTitle>
   <Journal><Title>J Two</Title><JournalIssue><PubDate><Year>2021</Year></PubDate></JournalIssue></Journal>
  </Article>
 </MedlineCitation>
</PubmedArticle>
<PubmedArticle>
 <MedlineCitation><PMID>333</PMID>
  <Article><ArticleTitle>Third</ArticleTitle>
   <Abstract><AbstractText>Only abstract.</AbstractText></Abstract>
   <Journal><Title>J Three</Title><JournalIssue><PubDate><MedlineDate>2019 Jan</MedlineDate></PubDate></JournalIssue></Journal>
  </Article>
 </MedlineCitation>
</PubmedArticle>
</PubmedArticleSet>"""


def test_parse_three_articles():
    recs = parse_pubmed_xml(FIXTURE)
    assert len(recs) == 3
    by_pmid = {r["pmid"]: r for r in recs}

    r1 = by_pmid["111"]
    assert r1["title"] == "First title"
    assert "BACKGROUND: Bg text." in r1["abstract"]
    assert "RESULTS: Res text." in r1["abstract"]
    assert r1["journal"] == "J One"
    assert r1["year"] == 2020
    assert r1["mesh_terms"] == ["Diabetes"]
    assert r1["parse_ok"] is True

    r2 = by_pmid["222"]
    assert r2["abstract"] is None  # missing abstract -> None, not dropped
    assert r2["year"] == 2021

    r3 = by_pmid["333"]
    assert r3["abstract"] == "Only abstract."
    assert r3["year"] is None  # MedlineDate without Year -> None


def test_parse_garbage_returns_empty():
    assert parse_pubmed_xml(b"not xml at all <<<") == []
