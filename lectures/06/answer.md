# 第6回 解答

> **実行環境について**: 課題は Codespaces ではなく、ローカルマシンの **DGX Spark（NVIDIA GB10）** で実行した。
> Python 環境は `~/.venv`（Python 3.12.3）。`uv`／`uvx`（`~/.local/bin`）を STEP 3 で使用（`uvx mcp-server-fetch==2026.11.11` を取得して起動）。
> API キーはリポジトリ直下の `.env`（gitignore 済み）から読み込んだ。生の出力原文は `lectures/06/work/` に保存してある。
> ブランチは `lec06`（第6回の更新は取り込み済み）。

## 演習

<!-- [tool call] の行は、表示された行をそのまま貼り付けてください。終わらなかった STEP は、空欄のままでかまいません。 -->

- STEP 1 のツールの名前：**`glossary_search_glossary`**（モデルに見えているツール: 1 個）。表示は `[server] glossary: コマンドで起動（uv run python -m app.mcp_server）` → ツール一覧で、引数は `{"query": {"title": "Query", "type": "string"}}`。比較した `app.agent --list-tools` は 4 ツール（`search_glossary` / `get_lecture_summary` / `make_quiz` / `search_materials`）で、`glossary_search_glossary` はこの `search_glossary` と**説明文・引数が同じ**だった（名前の先頭にサーバー名 `glossary_` が付くだけ）
- STEP 2 の [tool call]：`glossary_search_glossary({'query': '3レベル'})` → `[tool return] glossary_search_glossary -> [{"term": "宿題の3レベル", "lecture": 1, "definition": "AIE2026の宿題の構成。「基礎（講義内容の確認）」「実装（自分でコードを書く）」「発展（調査や試行）」の3段階があり、どれか1つに取り組めば…`
  - `[usage] 入力 715 tokens / 出力 152 tokens / モデル応答 2 回`（1 回目でツールを呼び、2 回目で結果を読んで回答。問題文の想定どおり）。回答は「基礎／実装／発展の 3 段階、どれか 1 つで完了、原則採点なし」で正しく、末尾に（関連する回：第1回）付き
- STEP 3 の [tool call]：`fetch_fetch({'url': 'https://modelcontextprotocol.io/'})`
  - `--list-tools` では `[server] fetch: コマンドで起動（uvx mcp-server-fetch==2026.11.11 --no-readability-js）` が増え、`fetch_fetch` が加わった（ツール計 2 個）。説明文は英語（"Fetches a URL from the internet and optionally extracts its contents as markdown. …"）で、引数は `url` のほか `max_length` / `start_index` / `raw` / `timeout_ms` が定義されていた
  - `[usage] 入力 2,179 tokens / 出力 117 tokens / モデル応答 2 回`。回答は MCP の 3 行要約（「AI アプリと外部データソース・ツールをつなぐオープンソースの標準規格」「AI にとっての USB-C ポートのような役割」等）
  - 実行ログに `GET https://modelcontextprotocol.io/robots.txt "HTTP/1.1 200 OK"` が 2 回出ており、**既定どおり robots.txt を確認してから取得**していた（補足の「既定の動作」の実例）
- STEP 4 の [tool call]：
  - `aws_aws___search_documentation({'search_phrase': 'Amazon S3 バージョニング 概要'})`
  - `aws_aws___search_documentation({'search_phrase': 'Amazon S3 バージョニングとは'})`
  - `aws_aws___search_documentation({'topics': ['general'], 'search_phrase': 'Using versioning in S3 buckets'})`
  - `--list-tools` では `[server] aws: URL に接続（https://knowledge-mcp.global.api.aws）` が増え、`aws_aws___read_documentation` / `aws_aws___search_documentation` / `aws_aws___list_regions` / `aws_aws___get_regional_availability` / `aws_aws___retrieve_skill` の 5 ツールが加わった（ツール計 7 個）。`command` のない `{"url": ...}` のみの設定
  - モデルは 2 回目で結果が足りないと判断し、検索語を英語・トピック指定に変えて**計 3 回呼び分けた**。`[usage] 入力 17,360 tokens / 出力 180 tokens / モデル応答 4 回`。回答は AWS 公式ドキュメントに基づく正しい説明（「同一バケット内でオブジェクトの複数バージョンを保持・取得・復元でき、誤った上書き・削除から回復できる」）
- 気づき・疑問：
  - **「自分の関数」から「MCP のツール」への変換は、名前の prefix 以外は無変換だった**。`glossary_search_glossary` の説明文・引数スキーマは `search_glossary` の docstring と型ヒントがそのまま MCP 経由で渡り、モデルから見るとローカル登録のツール（第3回の `app.agent`）と同じ形だった。これが「エージェントからサーバーが起動され、そのツールがモデルに渡る」ことの実感できたポイント
  - ほかの人が作ったサーバーは**設定 4 行（fetch）・1 行（aws）の追加だけで使えた**。ただし fetch の説明文は英語のまま（サーバー作者が書いたものが渡る）、AWS のツール名は `aws_aws___search_documentation` のようにサーバー名＋ツール名の内部構造まで含む名前になり、呼び方がサーバーごとに異なることをモデルは trace の情報から学んでいた
  - STEP 4 は「質問の内容が AWS に送られる」方式。実際、モデルが日本語の検索語をそのまま AWS に送って結果を得ていた（`search_phrase: 'Amazon S3 バージョニング 概要'`）。外部サーバーに送る情報の選択は使い手の責任、という補足の意味が分かった
  - fetch サーバーは robots.txt を見ていた。補足で読んだ「既定の動作（非公開アドレスを取得しない・robots.txt を尊重）」がログで確認できた
  - 疑問: `aws_aws___search_documentation` のように名前が二重になる（`aws_` + `aws___`）のは、AWS 側のツール名が `aws___search_documentation` のため。プレフィックス方式は名前が長くなりモデルが間違えやすい（実際、モデルは正しく呼べていたが）ので、クライアント側で名前を短くするオプションがあるとよい、と感じた


## 宿題

<!-- 基礎・実装・発展のどれか1つに取り組めば完了です。 -->

- 取り組んだレベル：**発展**（Claude Code の MCP サーバー設定方法を公式ドキュメントで調べ、`app/mcp_servers.json` と比較）
- やったこと：
  - **Claude Code** を選んだ（講義で扱ったハーネスの 1 つのため）。公式ドキュメントの「Connect Claude Code to tools via MCP」（リファレンス）と「Connect to MCP servers」（Quickstart）を読んだ
    - 参照 URL: `https://code.claude.com/docs/en/mcp`、`https://code.claude.com/docs/en/mcp-quickstart`
    - 調べた日付: **2026-11-17**（ドキュメントは更新が頻繁で、ページには Claude Code v2.1.3xx の記載もあった）
  - 調べた内容（4 項目）:

    | 調べること | Claude Code での方法 |
    |---|---|
    | 設定を書く場所 | 基本は **CLI コマンド `claude mcp add`**（`claude` セッション内ではなく、ターミナルから実行）。`claude mcp list` / `claude mcp get` / `claude mcp remove` で管理。ファイルで直接書く場合は、project scope なら**リポジトリ直下の `.mcp.json`**、local・user scope なら `~/.claude.json`。Quickstart には「`.mcp.json` はセッション開始時に読まれるので、編集したらセッションを再起動する」「起動時にプロジェクトのサーバーを使うか承認を求められ、`claude mcp reset-project-choices` で承認をやり直せる」という記載もあった |
    | コマンドで起動する書き方（"command" 相当） | `claude mcp add <name> -- <command> <args...>`（stdio サーバー）。`--env KEY=value` でサーバーに渡す環境変数を指定できる。`.mcp.json` では演習と同じ `mcpServers` 形式（`{"command": "uv", "args": [...]}`）で書ける。`claude mcp add-json` で JSON を渡してまとめて追加も可能。`Add a server from an npx, uvx, or binary command` のセクションもあり、`uvx` コマンドをそのまま登録できる |
    | URL で接続する書き方（"url" 相当） | `claude mcp add --transport http <name> <url>`（Streamable HTTP）。SSE は `--transport sse`、WebSocket も Option 4 として対応。`.mcp.json` では `{"type": "http", "url": "..."}` のように **`type` フィールドで方式を明示**する。リモートサーバーは OAuth サインイン（`/mcp` → Authenticate）や `--header` / 動的ヘッダー（環境変数から読むヘルパーコマンド）による認証もサポート |
    | 有効になる範囲 | **3 つの scope**: `local`（既定。そのユーザーがそのプロジェクトでだけ使える）、`project`（リポジトリの `.mcp.json` に書いて git でチーム共有。起動時に承認を求める）、`user`（そのユーザーのすべてのプロジェクトで使える）。`claude mcp add` に `--scope project` / `--scope user` を付けて選ぶ。同名サーバーが複数 scope にあるときの優先順位も定義されている |
  - 比較の軸として、演習の `app/mcp_servers.json`（最終形）を並べた:

    ```json
    {
      "mcpServers": {
        "glossary": {"command": "uv", "args": ["run", "python", "-m", "app.mcp_server"]},
        "fetch": {"command": "uvx", "args": ["mcp-server-fetch==2026.11.11", "--no-readability-js"]},
        "aws": {"url": "https://knowledge-mcp.global.api.aws"}
      }
    }
    ```
- 結果（演習の `app/mcp_servers.json` との比較）:
  - **JSON の形式そのものはほぼ同じ**だった。Claude Code の `.mcp.json` も `mcpServers` キーにサーバー名 → 設定の対応を書く形式で、`command` + `args`（コマンド起動）と `url`（URL 接続）という書き分けは演習と共通。演習の `mcp_servers.json` は「Claude Desktop や Cursor と同じ mcpServers 形式」であり、Claude Code の `.mcp.json` もこの系列。違いは ①URL 接続で **`type: "http"`（または `"sse"`）の明示が必要**（演習は `url` キーの有無だけで判別）②**認証の書き方が用意されている**（`env` フィールドのほか、OAuth フロー、`--header` / ヘルパーコマンドによる動的ヘッダー。AWS Knowledge のように認証不要なサーバーなら設定は演習と同程度で済む）③**環境変数の展開**（`${VAR}` を `.mcp.json` 内で書ける）
  - **設定の置き場所と配り方が最大の違い**。演習は「リポジトリに `app/mcp_servers.json` を置き、エージェントが起動時に読む」＝project scope 固定。Claude Code は CLI で scope を選べ、local（自分×このプロジェクト）/ project（`.mcp.json` を git で共有・起動時承認）/ user（自分の全プロジェクト）の 3 段階。チーム共有の手段も「ファイルをコミット」から「scope を選ぶ」に抽象化されている
  - **登録方法が「ファイル編集」から「CLI コマンド」が主流**だった点も違い。演習は JSON を手で編集する（STEP 3・4 でカンマの抜けに注意した方法）。Claude Code は `claude mcp add` が設定ファイルの書き込みを代行し、`claude mcp list` で接続状態（✔ Connected / tools fetch failed など）まで確認できる
- 気づき・疑問：
  - `mcpServers` 形式が **Claude Desktop・Cursor・Claude Code で共通言語になっている**ことが一番の発見だった。講座の `mcp_agent.py` は同じ形式を読む自前クライアントで、MCP の本来の価値（サーバーを 1 つ作ると、対応するすべてのクライアントから使える）をそのまま体験した。第3回の `agent.py`（関数を直接登録）と第6回の `mcp_agent.py`（サーバーから受け取る）の違いは「ツールの配布形式」の違いだった
  - Claude Code には演習にない運用機能が多い: 接続状態の確認（`claude mcp list`）、プロジェクトサーバーの承認、OAuth／ヘッダー認証、`.mcp.json` 内の環境変数展開、タイムアウト（`MCP_TIMEOUT`）設定。外部サーバー（特に URL 接続）を使う運用では、認証と範囲（scope）の管理が必須になる、という構図が読み取れた
  - 疑問: `aws_aws___search_documentation` のようにサーバー名の prefix が二重になる命名は、Claude Code でも同じなのか（サーバー名 `aws` × ツール名 `aws___search_documentation`）。また、演習の `load_mcp_servers()` は接続方式の自動判定（"auto"）を "legacy" に固定して警告を消しているが、Claude Code の CLI ではそのような問題が表面化しない設計（transport をコマンド側で明示させる）になっており、同じ問題の別解決法に見えた
