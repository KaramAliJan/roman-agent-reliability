# Final Policy Corpus — RomanAgent-Reliability

6 documents, 26 chunks, fully extracted, chunked, and retrieval-tested.
This replaces the earlier 3-document placeholder corpus.

## Documents and what's grounded in real law vs. authored

| Document | Sections | Basis |
|---|---|---|
| `leave_policy.pdf` | 6 | **Grounded in real Pakistani labor law**: 14 days annual leave (Factories Act 1934), 10 days casual leave, 16 days sick leave at half pay, 12 weeks maternity leave (Maternity and Paternity Leave Act). Paternity leave (5 days) is a common company norm, not a federal legal mandate — flagged here so you don't cite it as statutory in your paper. |
| `compensation_benefits_policy.pdf` | 4 | **EOBI figures grounded in real, current rates**: employer 5% / employee 1% of the government-notified minimum wage (PKR 40,000 as of FY 2025–26), per the EOBI Act 1976. Provident fund and medical insurance terms are authored, reasonable company norms. |
| `expense_policy.pdf` | 5 | Authored. PKR amounts are reasonable illustrative figures, not sourced from a real company. |
| `travel_policy.pdf` | 4 | Authored, except the 100Wh battery rule, which reflects real international air-travel carry-on regulations (not Pakistan-specific — this is a general aviation safety rule). |
| `it_support_policy.pdf` | 4 | Authored. |
| `meetings_policy.pdf` | 3 | Authored. |

**For your methodology section**: a fair and accurate line is something like
*"Leave entitlements and EOBI contribution rates are grounded in current
Pakistani labor law and statutory rates; all other policy content
(expense limits, IT procedures, meeting rules) was authored for this
project to avoid using any real company's proprietary material."*

## Review process (worth citing as your QC method)

1. Extracted text from each PDF with `pdfplumber`, chunked by numbered
   sub-section with a regex, saved to `final_corpus_chunks.json`
2. Printed every chunk with word counts and flagged any unusually short
   or long ones for manual review — none were flagged
3. Ran 10 test queries against the corpus and manually checked each
   retrieved chunk was actually correct
4. **Caught a real bug during this process**: the query "gym membership
   expense" initially matched the wrong chunk, because plain TF-IDF has
   no stemming — "membership" (query) and "memberships" (document) were
   treated as unrelated tokens. Fixed by adding Porter stemming to the
   tokenizer in `retrieval.py`. Re-ran all 10 queries after the fix; all
   resolved to the correct chunk.

This bug-and-fix is worth a sentence in your paper's methodology or
limitations section — it's a legitimate, citable point about TF-IDF's
weaknesses, and documenting that you caught and fixed it demonstrates
real QC rather than just asserting the corpus works.

## Known limitation worth flagging in your paper

Porter stemming fixes English morphology (membership/memberships) but is
English-specific — it does nothing for Roman Urdu queries. This means
your TF-IDF retriever's known weakness (no stemming) is only half-fixed:
solid for English variants, but Roman Urdu queries may hit the same class
of failure the stemming fix just solved for English. This is a legitimate
thing to test for and discuss in your results, not something to quietly
paper over — it may well BE one of your findings.

## Files
- `*.pdf` — the 6 source policy documents
- `final_corpus_chunks.json` — extracted, chunked corpus (26 chunks)
- `retrieval.py` — final retrieval module with stemming, drop-in
  replacement for the earlier placeholder version
- `process_final_corpus.py` — the extraction + chunking + verification
  script that produced `final_corpus_chunks.json` (re-run this if you
  add more policy PDFs later)

## Next step
Update your dataset's `expected_retrieval_doc_id` values to reference
these real chunk IDs (e.g. `leave_policy_sec1_2`, not a placeholder),
and write your `multi_hop_retrieval` and `negative_rejection` tasks
against this corpus specifically — `compensation_benefits_policy_sec6_4`
(remote work stipend, explicitly not covered) is ready to use as a
negative_rejection task as-is.
