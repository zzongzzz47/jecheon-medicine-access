"""Export the three Chapter 2 figures without their overall titles.

Reuse the original plotting code and saved analysis. Preserve panel labels,
legends, axes, and data; leave original figures and reports untouched.
"""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import build_resident_report as report


OUTPUT = report.OUT / "figures_report"
NAMES = {
    "01_현재와_10곳추가_거리지도": "그림2-1_수거망_접근거리_제목제거",
    "03_추가규모별_효과": "그림2-2_추가규모별_효과_제목제거",
    "04_10곳_선정위치": "그림2-3_대안별_선정위치_제목제거",
}


class FiguresComplete(Exception):
    pass


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    exported = []

    def save_figure(fig, name):
        if name in NAMES:
            if fig._suptitle is not None:
                fig._suptitle.remove()
                fig._suptitle = None
            path = OUTPUT / (NAMES[name] + ".png")
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
            exported.append(path)
            print(path)
        report.plt.close(fig)
        if name == "04_10곳_선정위치":
            # Stop before report generation or any analysis file updates.
            raise FiguresComplete

    original_save = report.savefig
    report.savefig = save_figure
    try:
        report.main()
    except FiguresComplete:
        pass
    finally:
        report.savefig = original_save
        report.plt.close("all")

    if len(exported) != len(NAMES):
        raise RuntimeError("Not all requested figures were exported")
    archive = OUTPUT / "제천_2장_그림_제목제거.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as bundle:
        for path in exported:
            bundle.write(path, arcname=path.name)
    print(archive)


if __name__ == "__main__":
    main()
