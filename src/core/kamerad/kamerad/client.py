import sys
import requests


def cli_main():
    endpoint = sys.argv[1]
    data = sys.argv[2]
    print(requests.post("http://localhost:8987/{}".format(endpoint), data=data).text)


if __name__ == "__main__":
    cli_main()
