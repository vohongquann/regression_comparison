# CLAUDE.md

Project nghiên cứu 9 thuật toán hồi quy. Bố cục và cách chạy: [README.md](README.md).

## Up lên Notion

Khi người dùng nói "up lên notion" (hoặc đồng nghĩa: đẩy / đồng bộ / sync lên Notion), dùng **notion-kit** để vừa
up vừa đồng bộ dữ liệu. Không viết script gọi API Notion riêng, không dùng MCP Notion.

* notion-kit ở `~/Documents/GitHub/notion-kit`, đã cài editable (`notion-kit` có sẵn trong PATH). Đọc
  `~/Documents/GitHub/notion-kit/CLAUDE.md` trước khi dùng. Khoá API nằm ở `~/.notion_api_key`, không in ra.
* Nguồn dữ liệu là `docs/` (9 bài + `Theo dõi chất lượng.md`). Nếu người dùng vừa sửa bài ở AI-Algorithm thì chạy
  `python scripts/sync_docs.py` trước, rồi mới up.
* "Đồng bộ" nghĩa là chạy lại được mà không tạo trùng: dùng `pages.upsert_page` (khớp theo tiêu đề) cùng `JsonCache`,
  rồi `cache.save()`. Bài đã có thì cập nhật, bài chưa có thì tạo. Không tạo trang mới mỗi lần.
* Chưa biết trang cha trên Notion thì hỏi một lần, rồi ghi `page_id` vào mục dưới để lần sau khỏi hỏi.
* Chỉ up khi người dùng bảo. Sửa file trong `docs/` hay `checks/` không tự kéo theo việc up.

Trang cha trên Notion: _(chưa ghi)_
