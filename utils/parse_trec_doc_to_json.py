import argparse
import json
import re
import xml.etree.ElementTree as ElementTree
from pathlib import Path

from tqdm import tqdm


class TREC45_Parser:
    def __init__(self, all_in_one=False):
        self.start = re.compile("<DOC>")
        self.end = re.compile("</DOC>")
        self.id_line = re.compile("<DOCNO>(?P<id>.*)?</DOCNO>")
        self.text_start = re.compile("<TEXT>")
        self.text_end = re.compile("</TEXT>")
        self.skip_pattern = None
        self.title_str = "HEADLINE"
        self.title_start = re.compile("<HEADLINE>")
        self.title_end = re.compile("</HEADLINE>")
        self.title_patern = re.compile("<HEADLINE>(?P<headline>.*)?</HEADLINE>")
        self.all_in_one = all_in_one

    def xml2json(self, input_file, output, skip_pattern=None):
        doc_flag = False
        with input_file.open(mode="r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if self.start.match(line):
                    doc_flag = True
                    doc = []

                if doc_flag:
                    doc.append(line)

                if doc_flag and self.end.match(line):
                    try:
                        try_doc = "".join(doc)
                        parsed_doc = ElementTree.fromstring(try_doc)
                        dline = self.parserd_xml2json(parsed_doc)
                    except ElementTree.ParseError:
                        # print("handmade")
                        dline = self.handmade_parser(doc)
                    except NotImplementedError:
                        dline = self.handmade_parser(doc)

                    if self.all_in_one:
                        new_dline = dict()
                        new_dline["id"] = dline["id"].strip()
                        new_dline["contents"] = ""
                        for k, v in dline.items():
                            if k == "id":
                                continue
                            new_dline["contents"] += v + "\n"
                            if "FT922-2482" in new_dline["id"]:
                                print(new_dline["contents"])

                        dline = new_dline

                    jline = json.dumps(dline)
                    print(jline, file=output)
                    doc_flag = False

    def handmade_parser(self, doc):
        text_container_flag = False
        title_flag = False
        title = ""
        for line in doc:
            if line and self.id_line.match(line):
                doc_no = self.id_line.match(line).group("id").strip()

            if self.title_start.search(line):
                if self.title_end.search(line):
                    title = self.title_pattern.search(line).group("title")
                else:
                    title_flag = True
                    title_container = []

            if title_flag:
                title_container.append(line)

            if self.title_end.match(line):
                title_flag = False
                title = "".join(title_container)

            if self.text_start.match(line):
                text_container = []
                text_container_flag = True
                continue

            if self.text_end.search(line):
                text_container_flag = False

            if text_container_flag:
                if self.skip_pattern:
                    if self.skip_pattern.match(line):
                        continue
                if line.strip():
                    text_container.append(line)

        dline = {"id": doc_no, "contents": " ".join(text_container), "title": title}
        return dline

    def parserd_xml2json(self, parsed_doc):
        line = dict()
        for child in parsed_doc.iter():
            if child.tag == "DOCNO":
                line["id"] = child.text.strip()

            if child.tag == "TEXT":
                line["contents"] = child.text.strip() + "\n"

            if child.tag == "DATE":
                line["date"] = child.text.strip()

            if child.tag == self.title_str:
                line["title"] = child.text.strip()

            """
            if child.tag == "PUB":
                line["publisher"] = child.text.strip()

            if child.tag == "PUB":
                line["publisher"] = child.text.strip()

            if child.tag == "PAGE":
                line["page"] = child.text.strip()
            """

        return line


class FR_Parser(TREC45_Parser):
    def __init__(self, all_in_one):
        super().__init__(all_in_one)
        self.skip_pattern = re.compile("<!\-*")

    def parserd_xml2json(self, parsed_doc):
        raise NotImplementedError


class FB_Parser(TREC45_Parser):
    def __init__(self, all_in_one):
        super().__init__(all_in_one)
        self.title_str = "TI"
        self.title_start = re.compile("<TI>")
        self.title_end = re.compile("</TI>")
        self.title_pattern = re.compile("<TI>(?P<title>.*)?</TI>")
        # self.skip_pattern = re.compile("<F.*</F>")


class LA_Parser(TREC45_Parser):
    def __init__(self, all_in_one):
        super().__init__(all_in_one)
        self.skip_pattern = re.compile("</?p>")

    def parserd_xml2json(self, parsed_doc):
        line = dict()
        text_flag = False
        title_flag = False
        date_flag = False
        for child in parsed_doc.iter():
            # print(child.tag)
            if child.tag == "DOCNO":
                line["id"] = child.text.strip()

            if child.tag == "TEXT":
                text_flag = True
                text = ""
                continue

            if text_flag and child.tag == "P":
                text += child.text.strip() + "\n"
                continue

            if text_flag:
                line["contents"] = text
                text_flag = False

            if child.tag == self.title_str:
                title_flag = True
                title = ""
                continue

            if title_flag and child.tag == "P":
                title += child.text.strip()
                continue

            if title_flag:
                line["title"] = title
                title_flag = False

            if child.tag == "DATE":
                date_flag = True
                date = ""
                continue

            if date_flag and child.tag == "P":
                date += child.text.strip()
                continue

            if date_flag:
                line["date"] = date
                date_flag = False

            """
            if child.tag == "PUB":
                line["publisher"] = child.text.strip()

            if child.tag == "PUB":
                line["publisher"] = child.text.strip()

            if child.tag == "PAGE":
                line["page"] = child.text.strip()
            """

        if text_flag:
            line["contents"] = text

        return line


def main(args):
    input_path = Path(args.input)
    output_path = Path(args.output_path)
    target = args.target
    all_in_one = args.all_in_one

    if target == "FT":
        input_files = sorted(input_path.glob("FT*/*"))
        parser = TREC45_Parser(all_in_one)
    elif target == "FR":
        input_files = sorted(input_path.glob("[0-9]*/*"))
        # input_files = sorted(input_path.glob("*"))
        parser = FR_Parser(all_in_one)
    elif target == "FB":
        input_files = sorted(input_path.glob("FB*"))
        parser = FB_Parser(all_in_one)
    elif target == "LA":
        input_files = sorted(input_path.glob("LA*"))
        parser = LA_Parser(all_in_one)
    else:
        raise ValueError(f"{target} doesn't exist")

    with output_path.open(mode="w") as output:
        for input_file in tqdm(input_files):
            print(input_file)
            if input_file.is_dir() or input_file.name == "MD5SUM":
                continue

            parser.xml2json(input_file, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", dest="input")
    parser.add_argument("-o", dest="output_path")
    parser.add_argument("-t", dest="target")
    parser.add_argument("-a", dest="all_in_one", type=bool, default=False)

    args = parser.parse_args()

    main(args)