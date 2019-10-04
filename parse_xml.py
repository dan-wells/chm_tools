#!/bin/python3

import argparse
import xml.etree.ElementTree as ET


def get_root(xmlfile):
    """Return the root element of an XML file.

    Args:
      xmlfile: Path to XML file
    Returns:
      root: xml.etree.ElementTree.Element containing the root of input XML

    Any namespaces are stripped from tag and attribute names to make subsequent
    parsing cleaner.
    """
    tree = ET.iterparse(xmlfile)
    for _, elem in tree:
        prefix, has_namespace, postfix = elem.tag.partition("}")
        if has_namespace:
            elem.tag = postfix
        for attrib in elem.attrib:
            prefix, has_namespace, postfix = attrib.partition("}")
            if has_namespace:
                elem.attrib[postfix] = elem.attrib.pop(attrib)
    root = tree.root
    return root


def parse_train_file(xmlfile, lang="arn"):
    """Extract tagged words under XML root element for target language."""
    root = get_root(xmlfile)
    words = [w for w in root.findall('./text/body/p/w')
                if w.attrib['lang'] == lang]
    for w in words[:5]:
        print(w.attrib)
        for m in w.findall('./m'):
            print(m.text, m.attrib)
    return words


def parse_test_file(xmlfile, lang="arn"):
    """Extract lines of raw text under XML root element for target language."""
    root = get_root(xmlfile)
    lines = [e.text for e in root.findall('./text/body/p')
                if e.attrib['lang'] == lang]
    for l in lines[:5]:
        print(l)
    return lines


def main():
    args = parse_args()
    if args.tagged:
        parse_train_file(args.xmlfile, args.lang)
    else:
        parse_test_file(args.xmlfile, args.lang)


def parse_args():
    """Parse script arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("xmlfile", type=str, help="Path to XML file to parse.")
    parser.add_argument("--lang", type=str, help="ISO 639-2/3 code for "
        "language to extract from XML file. Default arn = Mapuche.",
        default="arn", choices=("arn", "spa"))
    parser.add_argument("--tagged", action='store_true', help="Use to specify "
        "that input XML file has already been morphologically tagged.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
