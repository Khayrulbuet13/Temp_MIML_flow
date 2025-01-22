import os, datetime, time
repo_path = "/path/to/your/repo"
while True:
    os.chdir(repo_path)
    with open("update.txt", "a") as f:
        f.write(f"Update on {datetime.datetime.now()}\n")
    os.system("git add . && git commit -m 'Daily update' && git push")
    time.sleep(60 * 60 * 24)  # Wait 24 hours
