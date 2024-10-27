"""
Usage:
   python make_table_tasks.py --output <markdown_filename>
"""

import os
import sys
import json
import logging
import argparse

from pytablewriter import LatexTableWriter, MarkdownTableWriter


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_table(result_dict):
    """Generate table of results."""
    md_writer = MarkdownTableWriter()
    latex_writer = LatexTableWriter()
    md_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]
    latex_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]

    values = []

    for k, dic in sorted(result_dict["results"].items()):
        version = result_dict["versions"][k]
        percent = k == "squad2"
        for m, v in dic.items():
            if isinstance(v, str):
                logger.error(f"Result of {k}:{m} is not a number: {v}.")
                continue
            if m.endswith("_stderr"):
                continue

            if m + "_stderr" in dic:
                se = dic[m + "_stderr"]
                if percent or m == "ppl":
                    values.append([k, version, m, "%.2f" % v, "±", "%.2f" % se])
                else:
                    values.append(
                        [k, version, m, "%.2f" % (v * 100), "±", "%.2f" % (se * 100)]
                    )
            else:
                if percent or m == "ppl":
                    values.append([k, version, m, "%.2f" % v, "", ""])
                else:
                    values.append([k, version, m, "%.2f" % (v * 100), "", ""])
            k = ""
            version = ""
    md_writer.value_matrix = values
    latex_writer.value_matrix = values

    # todo: make latex table look good
    # print(latex_writer.dumps())

    return md_writer.dumps()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=argparse.FileType('r'), help="Summary json file.")
    parser.add_argument("--output", "-o", type=argparse.FileType('w'), default=sys.stdout, help="Output table file.")
    args = parser.parse_args()
    #
    print(args.summary)
    with args.summary as f:
        result_dict = json.load(f)
    with args.output as f:
        f.write(f"{make_table(result_dict)}\n")
