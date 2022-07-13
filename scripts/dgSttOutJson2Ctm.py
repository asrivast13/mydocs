import os
import io
import sys
import json
import argparse

#__MAIN__
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('jsonfiles', nargs="+", help="paths to JSON files to be converted to CTM")
    args = parser.parse_args()

    for file in args.jsonfiles:
        sessionId = os.path.basename(file).split('.')[0]
        try:
            with io.open(file, "r", encoding='utf8') as fp:
                data = json.load(fp)
            tokenList = data["results"]["channels"][0]["alternatives"][0]["words"]
            for token in tokenList:
                print("%s \t 1 \t %6.2f \t %4.2f \t %-15s \t %.2f" % (sessionId, token["start"], (token["end"]- token["start"]), token["word"], token["confidence"]))
        except Exception as e:
            print("\nERROR: Processing input JSON file: %s failed with error: %s" % (file, str(e)))
            sys.exit(-1)
