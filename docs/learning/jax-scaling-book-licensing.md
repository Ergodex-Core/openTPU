# JAX Scaling Book: source and redistribution review

Reviewed on 2026-10-23 UTC against official repository
[`jax-ml/scaling-book`, commit `173fb0851d7e654872f0327aff2e17607cf24106`](https://github.com/jax-ml/scaling-book/tree/173fb0851d7e654872f0327aff2e17607cf24106),
committed on 2026-10-19. The repository README identifies
[the official book website](https://jax-ml.github.io/scaling-book/).

**Decision: retain the download locally and publish an original linked guide and
download mechanism. Do not publish a complete copy of the book in OpenTPU.**
The upstream MIT grant is explicit, but a blanket redistribution grant for all
third-party figures was not established. This is a specific unresolved permission
question, not a claim that the book as a whole forbids redistribution.

## Evidence and scope

| Material | Evidence at the pinned revision | Treatment |
| --- | --- | --- |
| Repository's own software and documentation | [LICENSE](https://github.com/jax-ml/scaling-book/blob/173fb0851d7e654872f0327aff2e17607cf24106/LICENSE) is MIT; copyright 2022 Maruan Al-Shedivat and 2025 Google LLC. It permits copying, modification, and distribution subject to retaining the copyright and permission notices. | Preserve the complete upstream LICENSE in the local bundle. Do not relabel the book Apache-2.0 just because OpenTPU uses that license. |
| Character.ai figures `assets/img/kv-sharing.png` and `assets/img/prefix-caching-trie.png` | [inference.md](https://github.com/jax-ml/scaling-book/blob/173fb0851d7e654872f0327aff2e17607cf24106/inference.md) credits the [Character.ai optimizing-inference article](https://research.character.ai/optimizing-inference/). No separate reuse grant for these figures was found in the snapshot; that URL redirected to the blog homepage during review. | This is a concrete blocker to claiming complete public redistribution clearance. Local reading only; no images in this PR. |
| PagedAttention figure `assets/img/paged-attention.png` | The inference chapter credits [Kwon et al., arXiv:2309.06180](https://arxiv.org/abs/2309.06180). Its arXiv record links [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). | A separate attribution license exists; the repository MIT notice alone is not the complete notice for this source. The local source retains the credit. No figure is redistributed here. |
| JAX programming text and examples | [jax-stuff.md](https://github.com/jax-ml/scaling-book/blob/173fb0851d7e654872f0327aff2e17607cf24106/jax-stuff.md) says much of the section comes from the JAX shard-map document. | Preserve that source credit in the exact source/frontmatter. Any future public mirror must review the originating document's applicable license and notices too. |
| Site implementation and fonts | Embedded headers identify Distill JavaScript as Apache-2.0, Lit modules as BSD-3-Clause, Bootstrap as MIT, and Font Awesome/Scholar icons and fonts as CC BY 4.0 / SIL OFL 1.1 / MIT by component. The archive has only one standalone LICENSE file. | Do not assume the root MIT license supersedes those notices. Exclude site JavaScript, CSS, fonts, theme, packages, plugins, build files, and workflows from the reading bundle and PR. |
| Other externally credited figures, including GPU illustrations | Credits appear in chapter captions and surrounding text; source links alone do not establish a reuse grant. | Full image-by-image public redistribution clearance remains incomplete. No raster figures are committed. |

The audit inspected the repository license/README, chapter figure credits, file
inventory, and embedded asset headers. It is not a representation that every
third-party asset has been cleared. The downloader retrieves public files from
their existing upstream host for local study without bypassing access controls;
it neither sublicenses those files nor uploads them anywhere. A future public
mirror needs documented permissions, compatible licenses with complete notices,
or removal/replacement of affected material before publication.

## Attribution

The book's citation is Austin et al., *How to Scale Your Model*, Google DeepMind,
online, 2025. Its citation lists Jacob Austin, Sholto Douglas, Roy Frostig,
Anselm Levskaya, Charlie Chen, Sharad Vikram, Federico Lebron, Peter Choy,
Vinay Ramasesh, Albert Webson, and Reiner Pope. The upstream README credits
James Bradbury and Reiner Pope for foundational ideas, and al-folio and the
Distill team for the website theme. Preserve the upstream README and chapter
frontmatter rather than replacing them with this abbreviated attribution.

## What is and is not published

This PR contains original OpenTPU prose and Python code, source URLs, file names,
sizes, and SHA-256 values. It contains no copied chapter, image, font, third-party
script, model weight, or upstream build output. The local reading bundle contains
the original LICENSE and README plus exact chapter sources and referenced raster
figures, alongside explicitly labeled derived Markdown for local navigation.
The reader retains captions and credits. Interactive Plotly HTML is omitted and
linked to the official website; no downloaded executable code is run.

Return to the [reading guide and download/update instructions](jax-scaling-book.md).
