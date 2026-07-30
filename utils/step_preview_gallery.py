from __future__ import annotations

import argparse
import html
from pathlib import Path

from OCC.Display.WebGl.threejs_renderer import ThreejsRenderer
from OCC.Extend.DataExchange import read_step_file


STEP_SUFFIXES = {".step", ".stp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a browsable HTML gallery for STEP files."
    )
    parser.add_argument(
        "--steps-dir",
        type=Path,
        required=True,
        help="Directory that contains .step/.stp files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where preview pages will be written.",
    )
    parser.add_argument(
        "--mesh-quality",
        type=float,
        default=1.0,
        help="Tessellation quality passed to pythonocc. Larger is finer.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only render the first N STEP files.",
    )
    parser.add_argument(
        "--export-edges",
        action="store_true",
        help="Export visible edges into the preview page.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Suggested port to use with python -m http.server.",
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Only scan the top level of --steps-dir.",
    )
    parser.set_defaults(recursive=True)
    return parser.parse_args()


def collect_step_files(root: Path, recursive: bool) -> list[Path]:
    globber = root.rglob if recursive else root.glob
    files = [
        path
        for path in globber("*")
        if path.is_file() and path.suffix.lower() in STEP_SUFFIXES
    ]
    return sorted(files)


def preview_subdir(step_path: Path, steps_dir: Path) -> Path:
    rel = step_path.relative_to(steps_dir)
    suffix_name = rel.suffix.lower().lstrip(".")
    return rel.parent / f"{rel.stem}__{suffix_name}"


def patch_index_html(index_path: Path) -> None:
    text = index_path.read_text(encoding="utf-8")
    text = text.replace('src="/main.js"', 'src="./main.js"')
    index_path.write_text(text, encoding="utf-8")


def render_step_preview(
    step_path: Path,
    steps_dir: Path,
    output_dir: Path,
    export_edges: bool,
    mesh_quality: float,
) -> Path:
    target_dir = output_dir / preview_subdir(step_path, steps_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    shape = read_step_file(str(step_path))
    renderer = ThreejsRenderer(str(target_dir))
    renderer.DisplayShape(
        shape,
        export_edges=export_edges,
        mesh_quality=mesh_quality,
    )
    renderer.generate_html_file()
    patch_index_html(target_dir / "index.html")
    return target_dir / "index.html"


def build_gallery(output_dir: Path, generated: list[tuple[Path, Path]]) -> None:
    rows = []
    for step_path, preview_path in generated:
        step_label = html.escape(step_path.as_posix())
        rel_preview = preview_path.relative_to(output_dir).as_posix()
        rows.append(
            "<tr>"
            f"<td><a href=\"{html.escape(rel_preview)}\" target=\"_blank\">open</a></td>"
            f"<td>{step_label}</td>"
            "</tr>"
        )

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>STEP Preview Gallery</title>
  <style>
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f3f5f7;
      color: #16202a;
    }}
    main {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
    }}
    p {{
      margin: 0 0 16px;
      color: #4e5a65;
    }}
    input {{
      width: 100%;
      box-sizing: border-box;
      padding: 12px 14px;
      margin: 0 0 16px;
      border: 1px solid #cfd7df;
      border-radius: 10px;
      font-size: 15px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #ffffff;
      border-radius: 14px;
      overflow: hidden;
      box-shadow: 0 12px 32px rgba(9, 30, 66, 0.08);
    }}
    th, td {{
      padding: 12px 14px;
      border-bottom: 1px solid #e6ebf0;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #e9eef3;
      font-weight: 600;
    }}
    tr:last-child td {{
      border-bottom: none;
    }}
    a {{
      color: #0b63ce;
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
    .count {{
      font-weight: 600;
      color: #16202a;
    }}
  </style>
</head>
<body>
  <main>
    <h1>STEP Preview Gallery</h1>
    <p><span class="count">{len(generated)}</span> preview pages generated.</p>
    <input id="filter" type="search" placeholder="Filter by file name or path">
    <table>
      <thead>
        <tr>
          <th>Preview</th>
          <th>STEP file</th>
        </tr>
      </thead>
      <tbody id="rows">
        {"".join(rows)}
      </tbody>
    </table>
  </main>
  <script>
    const filter = document.getElementById("filter");
    const rows = Array.from(document.querySelectorAll("#rows tr"));
    filter.addEventListener("input", () => {{
      const keyword = filter.value.trim().toLowerCase();
      for (const row of rows) {{
        const text = row.textContent.toLowerCase();
        row.style.display = !keyword || text.includes(keyword) ? "" : "none";
      }}
    }});
  </script>
</body>
</html>
"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")


def main() -> int:
    args = parse_args()
    steps_dir = args.steps_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not steps_dir.is_dir():
        raise SystemExit(f"--steps-dir does not exist: {steps_dir}")

    step_files = collect_step_files(steps_dir, recursive=args.recursive)
    if args.limit is not None:
        step_files = step_files[: args.limit]

    if not step_files:
        raise SystemExit(f"No STEP files found under: {steps_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[tuple[Path, Path]] = []
    failures: list[tuple[Path, str]] = []

    total = len(step_files)
    for index, step_path in enumerate(step_files, start=1):
        rel = step_path.relative_to(steps_dir)
        print(f"[{index}/{total}] Rendering {rel}")
        try:
            preview_page = render_step_preview(
                step_path=step_path,
                steps_dir=steps_dir,
                output_dir=output_dir,
                export_edges=args.export_edges,
                mesh_quality=args.mesh_quality,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append((step_path, str(exc)))
            print(f"  FAILED: {exc}")
            continue
        generated.append((step_path, preview_page))

    if generated:
        build_gallery(output_dir, generated)

    if failures:
        failure_log = output_dir / "failures.txt"
        lines = [f"{path}\t{message}" for path, message in failures]
        failure_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Failures written to: {failure_log}")

    gallery_path = output_dir / "index.html"
    print()
    print(f"Generated {len(generated)} preview pages.")
    print(f"Gallery index: {gallery_path}")
    print("Serve it with:")
    print(f"  python -m http.server {args.port} --directory {output_dir}")
    print(f"Then open: http://127.0.0.1:{args.port}")

    return 0 if generated else 1


if __name__ == "__main__":
    raise SystemExit(main())
