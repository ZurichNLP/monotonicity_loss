#!/usr/bin/env python
import os
import re
import argparse

from lxml import etree


PARSER = etree.XMLParser(encoding='utf-8')

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f'readable_dir:{path} is not a valid folder path')

def parse_args():
    parser = argparse.ArgumentParser(description='Extract transliterated names by ids')
    parser.add_argument('--folder', type=dir_path, required=True, help='Folder containing XML files and id files.')
    return parser.parse_args()

def write_all_src_trg_pairs(name_node, src_file, trg_file):
    children = name_node.getchildren()
    source_name = re.sub(' ', u'⌀', children[0].text)
    for child in children[1:]:
        target_name = re.sub(' ', u'⌀', child.text)
        src_file.write(' '.join(source_name)+"\n")
        trg_file.write(' '.join(target_name)+"\n")

def write_src_only(name_node, src_file):
    children = name_node.getchildren()
    source_name = re.sub(' ', u'⌀', children[0].text)
    src_file.write(' '.join(source_name)+"\n")

def main():
    args = parse_args()

    # Load original training data - we split this and use as train and dev set
    train_dev_xml = etree.parse(os.path.join(args.folder, 'train.xml'), parser=PARSER)
    root = train_dev_xml.getroot()

    # Load name ids in our train and dev sets
    with open(os.path.join(args.folder, 'train.ids')) as id_file:
        train_ids = [name_id.strip() for name_id in id_file.readlines()]

    with open(os.path.join(args.folder, 'dev.ids')) as id_file:
        dev_ids = [name_id.strip() for name_id in id_file.readlines()]

    # Write examples to train and dev files
    with open(os.path.join(args.folder, 'train.src'), 'w') as train_src_file:
        with open(os.path.join(args.folder, 'train.trg'), 'w') as train_trg_file:
            with open(os.path.join(args.folder, 'dev-for-train.src'), 'w') as dev_src_file:
                with open(os.path.join(args.folder, 'dev-for-train.trg'), 'w') as dev_trg_file:
                    with open(os.path.join(args.folder, 'dev-for-eval.src'), 'w') as dev_src_eval_file:
                        for name_node in train_dev_xml.findall('Name'):
                            name_id = name_node.get('ID')
                            if name_id in train_ids:
                                write_all_src_trg_pairs(name_node, train_src_file, train_trg_file)
                                root.remove(name_node)
                            elif name_id in dev_ids:
                                # During training calculate cross-entropy on all possible src-trg pairs
                                write_all_src_trg_pairs(name_node, dev_src_file, dev_trg_file)
                                # For evaluation after training, only extract src name once
                                write_src_only(name_node, dev_src_eval_file)

    # Save dev split XML for evaluation
    train_dev_xml.write(os.path.join(args.folder, 'dev-for-eval.ref'), pretty_print=True, xml_declaration=True, encoding="utf-8")

    # Load original development data - we use this as the test set
    test_xml = etree.parse(os.path.join(args.folder, 'dev.xml'), parser=PARSER)

    # Write examples to test files
    with open(os.path.join(args.folder, 'test-for-scoring.src'), 'w') as test_src_file:
        with open(os.path.join(args.folder, 'test-for-scoring.trg'), 'w') as test_trg_file:
            with open(os.path.join(args.folder, 'test-for-eval.src'), 'w') as test_src_eval_file:
                for name_node in test_xml.findall('Name'):
                    # Monotonicity scoring on all possible src-trg pairs
                    write_all_src_trg_pairs(name_node, test_src_file, test_trg_file)
                    # For evaluation after training, only extract src name once
                    write_src_only(name_node, test_src_eval_file)

    # Save test split XML for evaluation
    test_xml.write(os.path.join(args.folder, 'test-for-eval.ref'), pretty_print=True, xml_declaration=True, encoding="utf-8")


if __name__ == '__main__':
    main()
