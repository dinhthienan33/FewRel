<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# tôi định chuyển sang dataset [https://aclanthology.org/D18-1514.pdf](https://aclanthology.org/D18-1514.pdf) -> lên plan thiết kế db và train, vẫn giữ nguyên pipeline graph + moe

Link bạn đưa là **D18-1514 = FewRel**, không phải DocRED; vì vậy nếu chuyển sang đúng dataset ở link đó thì bài toán sẽ đổi từ document-level RE sang few-shot sentence-level relation classification, và phần thiết kế DB/train cần tách rõ “raw schema của FewRel” với “adapter để vẫn chạy graph + MoE pipeline”. Nếu mục tiêu của bạn là “giữ nguyên pipeline graph + moe” ở mức tối đa, hướng hợp lý là giữ backbone kiến trúc nhưng thay graph từ document graph sang support-query prototype graph / relation graph, còn DB thì tổ chức theo **episode-centric** thay vì **document-centric**.[^1_1][^1_2]

## Đổi bài toán

FewRel là tập dữ liệu supervised few-shot relation classification, được xây để đánh giá few-shot RE; mỗi instance là một câu cùng head entity, tail entity và relation label, thay vì document nhiều câu với entity/relation ở mức tài liệu như DocRED. Vì vậy, các module phụ thuộc mạnh vào cross-sentence evidence, mention aggregation theo document, hay document-level adjacency sẽ không còn là trung tâm; thay vào đó, phần quan trọng là episode sampling, support-query split và relation prototype reasoning.[^1_2][^1_1]

## DB plan

Mình khuyên thiết kế DB 3 tầng: `raw_dataset`, `canonical_ie`, và `training_episodes` để vừa trace được dữ liệu gốc FewRel vừa không phá pipeline cũ.[^1_1][^1_2]

- `raw_dataset.relations(relation_id, relation_name, description, source='fewrel')`: lưu relation classes từ FewRel.[^1_2][^1_1]
- `raw_dataset.instances(instance_id, relation_id, tokens_json, sentence_text, head_span, tail_span, head_text, tail_text, meta_json)`: mỗi sample là một câu có cặp thực thể head-tail.[^1_1][^1_2]
- `canonical_ie.documents(doc_id, text, source, granularity='sentence-as-document')`: bọc mỗi sentence thành một “pseudo-document” để tái dùng pipeline graph cũ.[^1_2][^1_1]
- `canonical_ie.entities(entity_id, doc_id, entity_text, entity_type nullable, start_tok, end_tok, role in {head,tail})`: map head/tail sang entity nodes.[^1_1][^1_2]
- `canonical_ie.relations(rel_inst_id, doc_id, head_entity_id, tail_entity_id, relation_id, label, evidence_json)`: evidence lúc này chỉ là local sentence span.[^1_2][^1_1]
- `training_episodes(episode_id, N, K, Q, split, seed, sampled_relations_json)`: bảng trung tâm để tái lập few-shot episodes.[^1_1][^1_2]
- `training_episode_items(episode_id, relation_id, instance_id, item_role in {support,query})`: lưu membership của từng sample trong episode.[^1_2][^1_1]

Điểm mấu chốt là **canonical layer** giúp code graph hiện tại không cần rewrite toàn bộ: mỗi câu được coi như một document cực ngắn, graph vẫn có token/entity/relation nodes nhưng edges đơn giản hơn nhiều.[^1_1][^1_2]

## Graph + MoE adapter

Để giữ nguyên tinh thần graph + MoE, bạn có thể đổi graph thành 3 lớp node: token nodes, entity nodes, và relation/prototype nodes. Với FewRel, graph hợp lý nhất là:[^1_1]

- Intra-instance graph: token-token dependency/adjacency, token-entity, head-tail shortcut.
- Episode graph: support instance nodes nối với relation prototype node tương ứng.
- Query-support graph: query entity pair nối tới top-k support samples theo similarity.

MoE nên đặt ở chỗ fusion thay vì chỉ encoder cuối:

- Expert 1: lexical/context expert, mạnh cho local context sentence.
- Expert 2: entity-pair structure expert, chỉ nhìn head-tail markers, relative positions, dependency/path.
- Expert 3: support-matching expert, học so khớp query với support set.
- Expert 4: relation-prior expert, dùng relation description/name embedding nếu bạn có textual relation metadata.[^1_2][^1_1]

Gate có thể conditioning trên `query_repr + support_proto + graph_stats` để chọn expert theo từng query, giữ đúng tinh thần MoE nhưng phù hợp few-shot setting.[^1_2][^1_1]

## Train plan

Train nên chuyển từ document minibatch sang **episodic training**, ví dụ $N$-way $K$-shot với query set cố định mỗi episode, đúng với bản chất của FewRel. Loss nên là tổ hợp: classification loss trên query, prototype contrastive loss giữa relation classes, và auxiliary gating balance loss để tránh MoE collapse.[^1_1][^1_2]

Một plan triển khai gọn:

1. Data adapter: convert FewRel raw sample -> canonical sentence-document record + entity pair graph.[^1_2][^1_1]
2. Episode sampler: sinh `N-way K-shot Q-query`, cache vào bảng `training_episodes` để reproducible.[^1_1][^1_2]
3. Encoder: giữ text encoder cũ nếu được, nhưng thay document graph encoder bằng sentence/entity-pair graph encoder.[^1_2][^1_1]
4. Prototype builder: aggregate support embeddings theo relation để tạo class prototype; có thể dùng graph pooling thay mean-pooling thuần.[^1_1][^1_2]
5. MoE matcher: query qua nhiều expert rồi gate để ra query-to-prototype score.[^1_2][^1_1]
6. Evaluation: report chuẩn few-shot theo 5-way/10-way, 1-shot/5-shot thay vì F1 kiểu DocRED.[^1_1][^1_2]

## Khuyến nghị thực tế

Nếu bạn chỉ muốn “đổi dataset nhưng giữ gần như nguyên bài toán DOCIE/DocRE”, thì link này chưa hợp vì FewRel là sentence-level few-shot RE, lệch khá xa document-level extraction. Còn nếu bạn chấp nhận xoay paper sang hướng “graph + MoE for few-shot relation classification”, thì đây lại là cơ hội tốt để viết một adapter framework kiểu: **document-centric pipeline generalized to episode-centric relation reasoning**.[^1_2][^1_1]

Mình nghi bạn đang nhắm DocRED nhưng paste nhầm link. Nếu đúng là bạn muốn chuyển sang **FewRel**, mình có thể viết tiếp ngay bản rất cụ thể gồm: sơ đồ DB dạng SQL, schema JSON cho episode cache, và skeleton code PyTorch/HF để cắm graph + MoE vào training loop. Còn nếu bạn muốn chuyển sang **DocRED**, mình sẽ lên lại plan đúng theo document-level setting.
<span style="display:none">[^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_3][^1_4][^1_5][^1_6][^1_7][^1_8][^1_9]</span>

<div align="center">⁂</div>

[^1_1]: https://aclanthology.org/anthology-files/anthology-files/pdf/D/D18/D18-1514.pdf

[^1_2]: https://thunlp.github.io/1/fewrel1.html

[^1_3]: DOCIE_REPORT.pdf

[^1_4]: 2507.22926v1.pdf

[^1_5]: 2407.21384v2.pdf

[^1_6]: 2024.futured-1.6.pdf

[^1_7]: 2203.02721v2.pdf

[^1_8]: 2024.acl-demos.20.pdf

[^1_9]: https://huggingface.co/datasets/thunlp/docred

[^1_10]: https://arxiv.org/html/2505.09792v1

[^1_11]: https://arxiv.org/abs/2303.11141

[^1_12]: https://hal.science/hal-04064170/document

[^1_13]: https://www.scribd.com/document/683578715/docvredpape

[^1_14]: https://dl.acm.org/doi/10.1145/3477495.3531831

[^1_15]: https://arxiv.org/pdf/2008.12283v1.pdf

[^1_16]: https://arxiv.org/html/2407.01026v1

[^1_17]: https://aclanthology.org/2026.eacl-long.213.pdf

[^1_18]: https://dl.acm.org/doi/10.1145/3539618.3591911

[^1_19]: https://www.sciencedirect.com/science/article/pii/S2452414X21000959

[^1_20]: https://dl.acm.org/doi/10.1145/3539618.3591912


---

# phân tích tất tần tật data fewrel cho tôi

FewRel là một bộ dữ liệu few-shot relation classification ở mức câu, gồm ~70k câu và 100 quan hệ, được xây từ Wikipedia và Wikidata và annotate bởi crowdworker, với cấu trúc JSON khá “sạch” để dựng episode N-way K-shot cho huấn luyện few-shot RE. Dưới đây là phân tích “tất tần tật” theo góc nhìn mình nghĩ bạn cần cho thiết kế DB + pipeline DOCIE.[^2_1][^2_2][^2_3][^2_4]

## Tổng quan \& mục tiêu dữ liệu

- Nhiệm vụ: few-shot relation classification giữa một cặp thực thể (head, tail) trong một câu, tức xác định quan hệ $r$ giữa $h$ và $t$.[^2_3]
- Nguồn: câu được lấy từ Wikipedia với distant supervision (align Wikidata triples với câu) rồi được lọc / kiểm tra bởi crowdworkers để loại noise.[^2_3]
- Quy mô: khoảng 70,000 câu cho 100 relations, nghĩa là trung bình ~700 instance mỗi relation (train+val), nên đủ để tạo nhiều episode few-shot.[^2_5][^2_3]
- Ngôn ngữ: tiếng Anh, domain rộng (Wikipedia + các tập validation khác như NYT, PubMed, SemEval) giúp mô hình không overfit vào một domain nhỏ.[^2_4]

Ý nghĩa với DOCIE: đây là bộ dữ liệu hoàn toàn sentence-level, không phải document-level như DocRED, nhưng lại rất phù hợp để đánh giá cơ chế graph + MoE trong **few-shot** bối cảnh, vì số lượng class nhiều và data mỗi class ít khi bước vào unseen relations.[^2_1][^2_3]

## Cấu trúc file \& schema logic

Có hai phiên bản: FewRel 1.0 (paper D18-1514) và FewRel 2.0 (challenges về domain shift và noise), nhưng cấu trúc instance khá giống nhau.[^2_6][^2_3]

### Dạng JSON chuẩn (FewRel-style)

Một file JSON cho train thường là dictionary mapping từ tên quan hệ → list các instance:[^2_7]

- Key: tên relation (thường là Wikidata property, ví dụ `P26` cho “spouse”) hoặc tên dạng đọc được.[^2_7]
- Value: list instance, mỗi instance là một dictionary với các field chính:[^2_4][^2_7]
    - `tokens`: danh sách token của câu, dạng chuỗi.[^2_4]
    - `h`: “head entity”, thường là list với:
        - `[^2_0]`: text của entity (lowercase).[^2_7]
        - `[^2_1]`: Wikidata ID của entity (đôi khi rỗng nếu không linking).[^2_7]
        - `[^2_2]`: list chứa nested list các indices token thuộc entity mention (ví dụ `[[3,4]]`).[^2_7]
    - `t`: “tail entity”, cấu trúc tương tự `h`.[^2_7]
    - Một số biến thể thêm `ner`: BIO/BIOES tag cho mỗi token để lưu entity type (PER/ORG/LOC, …), hoặc `head_type`, `tail_type`.[^2_4][^2_7]

HuggingFace phiên bản `thunlp/few_rel` dùng schema gần giống nhưng “flatten” hơn:[^2_4]

- Field `relation`: string, tên relation.[^2_4]
- `tokens`: sequence string, câu đã tokenized.[^2_4]
- `head`: object với `text`, `type`, `indices` (một sequence nested indices).[^2_4]
- `tail`: tương tự head.[^2_4]
- Ngoài ra có split `pid2name`, một bảng map giữa relation id và tên relation.[^2_4]


### Các split \& domain

Trong TFDS/HF version, splits được tổ chức theo corpus/domain:[^2_4]

- `train_wiki`: ~44,800 instance train từ Wikipedia.[^2_4]
- `val_wiki`: ~11,200 instance validation từ Wikipedia.[^2_4]
- `val_nyt`: ~2,500 instance từ New York Times.[^2_4]
- `val_pubmed`: ~1,000 instance từ PubMed (y khoa).[^2_4]
- `val_semeval`: ~8,851 instance từ SemEval RE benchmark.[^2_4]
- `pubmed_unsupervised`: ~2,500 instance không label (dùng cho unsupervised/transfer).[^2_4]

Điều này quan trọng nếu bạn muốn đánh giá robustness/transfer: có thể train episode từ `train_wiki`, nhưng đánh giá trên `val_pubmed`/`val_nyt` để đo domain shift.[^2_6][^2_4]

## Thống kê chi tiết: quan hệ, instance, lengths

### Số quan hệ \& phân bố

Paper FewRel nói rõ: có 100 relations; mỗi relation chọn từ Wikidata property, được đảm bảo có đủ instance để few-shot.[^2_3]

- Tổng quan: 100 relation, mỗi relation ~700 câu (train+val), nhưng không hoàn toàn đều; vài relation như “country of citizenship”, “spouse”, “father”, “location of formation” có nhiều instance hơn vì phổ biến trên Wikipedia.[^2_3][^2_7]
- Các relation đến từ nhiều loại: nhân sự (father/mother/spouse/sibling), quốc gia, địa điểm, nghề nghiệp, tổ chức, địa lý (river, mountain range, tributary), v.v..[^2_7]

Đối với thiết kế MoE, bạn có thể nhóm relation theo meta-type (person-person, person-organization, location-location, organization-location, etc.) để tạo **expert theo cluster quan hệ**, hoặc dùng meta-type như feature cho gating.[^2_3][^2_7]

### Độ dài câu \& entity span

Câu trong FewRel đa phần là câu Wikipedia, nên:

- Length tokens: thường 15–40 tokens; có một số câu dài hơn (có mệnh đề phụ, liệt kê).[^2_3]
- Entity span indices: đa phần contiguous (liên tiếp) và ngắn, như `[[3,3]]` hoặc `[[5,7]]`, nhưng vẫn có multi-span (tên dài, có dấu phẩy).[^2_7]
- Cặp head-tail có thể xuất hiện *multiple times* trong câu (multi-mention), nhưng indices trong FewRel thường chỉ đánh dấu một span chính hoặc tất cả spans; bạn cần kiểm logic khi build graph (ví dụ: tạo node entity và edge đến tất cả span nodes).[^2_7][^2_4]

Từ góc độ graph, bạn có thể coi:

- Nodes: token nodes, entity nodes (head/tail), relation node.
- Edges: token adjacency, dependency edges (nếu bạn parse), entity-to-token edges, head-tail direct edge.

Do câu ngắn, graph khá nhỏ, nên overhead GNN thấp, phù hợp chạy nhiều episode.

### Label quality \& noise

FewRel sử dụng distant supervision + crowdworker filtering:[^2_3]

- Pipeline: từ Wikidata triple $(e_h, r, e_t)$, tìm câu chứa cả hai entity, gán label, sau đó crowdworkers kiểm tra xem câu thật sự biểu hiện quan hệ đó hay không.[^2_3]
- Kết quả: nhiều noise từ distant supervision (câu chỉ chứa hai entity nhưng không nói rõ relation) được loại bỏ; dataset final có label tương đối chuẩn, nhưng vẫn có case cần reasoning không-trivial.[^2_3]
- Paper báo cáo rằng best few-shot models vẫn còn kém xa human performance (human ~92% accuracy trong 5-way 1-shot, model thấp hơn), cho thấy câu nhiều khi khó hoặc subtle.[^2_8][^2_3]

Điều này hữu ích cho MoE: bạn có thể thêm expert chuyên reason dựa trên long-range syntax hoặc inference, không chỉ pattern matching đơn giản.

## Episode-level design: N-way K-shot

FewRel được xây để đánh giá few-shot theo các setting 5-way/10-way, 1-shot/5-shot:[^2_8][^2_3]

- 5-way 1-shot: 5 relations, mỗi relation 1 support example, và nhiều query example; human ~92.22% accuracy, SOTA models ~ >90%.[^2_8]
- 5-way 5-shot, 10-way 1-shot, 10-way 5-shot: các cấu hình khác với số relation/shot thay đổi; table benchmark hiện có nhiều model, từ Matching Networks đến prompt-based.[^2_9][^2_8]

Struct logic cho episode:

- Chọn N relations từ set 100.
- Cho mỗi relation, chọn K support instances và Q query instances (thường Q=10 hoặc hơn).[^2_3]
- Episode dictionary: `support_set` list, `query_set` list, mỗi phần tử là instance; maintain mapping `relation name → label index` trong episode.[^2_3]

Đối với DB, bạn nên cache thông tin episode (N,K,Q, seed, danh sách instance id) để reproducible và phân tích ablation.

## Domain \& generalization khía cạnh

Các split cho phép kiểm tra:

- **In-domain** (Wiki→Wiki): train_wiki, validate trên val_wiki.[^2_4]
- **Cross-domain**: train_wiki → validate trên val_nyt/val_pubmed/val_semeval để đo domain shift và transfer khả năng.[^2_6][^2_4]
- FewRel 2.0 được đề xuất để tạo thách thức hơn, thêm noise và domain transfer; paper nói rõ rằng FewRel 1.0 tương đối “clean” nhưng distribution test có thể unrealistic cho vài scenario, nên 2.0 điều chỉnh.[^2_10][^2_6]

Cho DOCIE, đây là chỗ để bạn design thí nghiệm: cùng pipeline graph + MoE, bạn có thể so sánh in-domain vs cross-domain để chứng minh khả năng generalize.

## Metadata: relation descriptions \& pid2name

HF TFDS mô tả thêm split `pid2name`, mapping:[^2_4]

- `relation`: string id quan hệ.[^2_4]
- `names`: sequence string, tên/alias của relation.[^2_4]

Wikidata property ID (Pxx) cung cấp metadata như:

- Label (short name).
- Description (ngắn giải thích).
- Domain/type constraint (ví dụ citizen-of expects person, country-of expects location).

Bạn có thể dùng metadata này làm textual prompt hoặc embedding cho relation node trong graph, hoặc dùng MoE expert chuyên xử lý textual semantics của relation.

## Hạn chế \& điểm cần cẩn trọng

- **Sentence-level, không có document context**: model không cần cross-sentence reasoning, nên mọi component của pipeline DOCIE dành cho document graph, evidence selection giữa câu… sẽ không phát huy hết.[^2_3]
- **Không có NONE class trong base FewRel**: dataset cơ bản gồm instance với relation đã biết, không có negative “no relation” class; nhiều work sau phải tự xây NONE/NOTA để realistic hơn. Nếu bạn muốn đánh giá detection, cần tạo negative sampling.[^2_11][^2_10]
- **Distribution test không hoàn toàn realistic**: paper “Towards Realistic Few-Shot RE” nêu rằng cách tạo episode trong FewRel 1.0 có thể lạc quan hơn thực tế và đề xuất cách xây data từ TACRED với NOTA; đây là điểm nếu bạn muốn làm paper về “realistic DOCIE few-shot”.[^2_11][^2_10]
- **Wikidata linking không luôn đầy đủ**: một số instance bỏ trống entity ID, nên nếu bạn muốn nối vào KB/global graph thì phải xử lý missing.[^2_7][^2_4]


## Gợi ý mapping sang thiết kế DB/pipeline

Từ toàn bộ phân tích trên, bạn có thể tổ chức schema như:

- `relations(pid, name, aliases, description, domain_meta)` lấy từ pid2name + Wikidata.[^2_4]
- `instances(id, split, relation_pid, tokens, head_text, head_indices, tail_text, tail_indices, ner_tags, source_corpus)`.[^2_7][^2_4]
- `episodes(id, N, K, Q, split, seed)` + `episode_items(id, episode_id, instance_id, role)`.[^2_8][^2_3]

Trong pipeline:

- Graph layer: node = tokens + head/tail entity + relation prototype; edges = adjacency, dependency, entity-token, entity-entity.
- MoE: chuyên xử lý các view khác nhau (lexical, syntactic, semantic, metadata) trên cùng episode.

Nếu bạn muốn, bước tiếp mình có thể:

- Viết cụ thể SQL/JSON schema cho từng table.
- Propose canonical graph representation và MoE expert design sao cho khớp “tất tần tật” đặc trưng FewRel mà vẫn giữ flavor DOCIE paper bạn đang viết.

Bạn có muốn mình tập trung phân tích thêm vào phần **episode construction** (cách sampling, balancing, realistic NOTA) hay **relation taxonomy + metadata** (cách dùng Wikidata để enrich graph/MoE)?
<span style="display:none">[^2_12][^2_13][^2_14][^2_15]</span>

<div align="center">⁂</div>

[^2_1]: https://github.com/thunlp/FewRel

[^2_2]: https://huggingface.co/datasets/thunlp/few_rel

[^2_3]: https://arxiv.org/abs/1810.10147

[^2_4]: https://www.tensorflow.org/datasets/community_catalog/huggingface/few_rel?hl=zh-cn

[^2_5]: https://github.com/gaotianyu1350/new_fewrel_bertpair

[^2_6]: https://www.arxiv.org/pdf/1910.07124.pdf

[^2_7]: https://github.com/acidAnn/fewrelde

[^2_8]: https://thunlp.github.io/1/fewrel1.html

[^2_9]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12840248/table/entropy-28-00069-t001/

[^2_10]: https://aclanthology.org/2021.emnlp-main.433.pdf

[^2_11]: https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00392/106791/Revisiting-Few-shot-Relation-Classification

[^2_12]: https://aclanthology.org/anthology-files/anthology-files/pdf/D/D18/D18-1514.pdf

[^2_13]: https://metatext.io/datasets/fewrel-1.0

[^2_14]: https://dl.acm.org/doi/pdf/10.1145/3447548.3467438

[^2_15]: https://arxiv.org/pdf/2210.08242.pdf

