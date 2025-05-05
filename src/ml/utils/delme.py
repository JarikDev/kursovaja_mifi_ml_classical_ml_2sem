import requests
resp =  requests.get("https://colab.research.google.com/drive/1tUFQ8WfYarruudNW5yCZIYgVQOSjk1eH#scrollTo=QmCBqQPpQCtw")
print(resp.status_code)
resp =  requests.get("https://colab.research.google.com/drive/1Ij8EIRAckgQJAgSFpye9ch1bdul4ufP8")
print(resp.status_code)