"""Hand-written evaluation set for data/doc_40.pdf (Pro Git, chapters 1-2).

Each item:
  q        - the question asked
  evidence - short phrases that appear ONLY in the passage that answers it.
             A retrieved chunk counts as a hit if it contains any of them.
  answer   - keywords; the LLM answer counts as correct if it contains any of them.

Matching is case-insensitive, with curly quotes/apostrophes and whitespace normalised.
"""
import re

QA = [
    {"q": "In what year did the Linux kernel project start using BitKeeper?",
     "evidence": ["began using a proprietary DVCS"], "answer": ["2002"]},
    {"q": "Why did the Linux community stop using BitKeeper?",
     "evidence": ["free-of-charge status was revoked"], "answer": ["revoked", "free-of-charge", "free of charge"]},
    {"q": "How does Git store data differently from delta-based version control systems?",
     "evidence": ["takes a picture of what all your files look like"], "answer": ["snapshot", "picture"]},
    {"q": "What mechanism does Git use for checksumming?",
     "evidence": ["mechanism that git uses for this checksumming"], "answer": ["sha-1", "sha1"]},
    {"q": "What are the three main states that files can reside in within Git?",
     "evidence": ["modified, staged, and committed"], "answer": ["modified, staged, and committed", "staged"]},
    {"q": "What is the technical name of the staging area in Git?",
     "evidence": ["technical name in git parlance"], "answer": ["index"]},
    {"q": "Which command installs Git on Fedora?",
     "evidence": ["such as rhel or centos), you can use dnf"], "answer": ["dnf"]},
    {"q": "Where can you download the macOS Git installer?",
     "evidence": ["macos git installer is maintained"], "answer": ["git-scm"]},
    {"q": "How do you set your user name for all repositories?",
     "evidence": ["git config --global user.name"], "answer": ["--global user.name", "user.name"]},
    {"q": "What branch name does git init create by default?",
     "evidence": ["create a branch called master"], "answer": ["master"]},
    {"q": "From which Git version can you set a different name for the initial branch?",
     "evidence": ["from git version 2.28 onwards"], "answer": ["2.28"]},
    {"q": "How do you get the more concise help output for a Git command?",
     "evidence": ["more concise \"help\" output"], "answer": ["-h"]},
    {"q": "What data does Git receive when you clone a repository?",
     "evidence": ["every version of every file for the history"], "answer": ["every version", "full copy", "all data"]},
    {"q": "What does a URL using the SSH transfer protocol look like?",
     "evidence": ["user@server:path/to/repo.git"], "answer": ["user@server"]},
    {"q": "What two states can each file in your working directory be in?",
     "evidence": ["can be in one of two states: tracked or untracked"], "answer": ["untracked"]},
    {"q": "In short git status output, what marks new files that aren't tracked?",
     "evidence": ["have a ?? next to them"], "answer": ["??"]},
    {"q": "What files does the .gitignore pattern *.[oa] ignore?",
     "evidence": ["object and archive files"], "answer": ["object", "archive", ".o"]},
    {"q": "How do you match nested directories in a .gitignore pattern?",
     "evidence": ["two asterisks to match nested directories"], "answer": ["two asterisks", "**"]},
    {"q": "Which git diff option shows the changes you have staged?",
     "evidence": ["--staged and --cached are synonyms", "git diff --staged"], "answer": ["--staged", "--cached"]},
    {"q": "How can you view diffs in an external tool such as vimdiff?",
     "evidence": ["if you run git difftool instead of git diff"], "answer": ["difftool"]},
    {"q": "How do you remove a file from the staging area but keep it on your hard drive?",
     "evidence": ["git rm --cached readme"], "answer": ["--cached"]},
    {"q": "Which git log option shows the patch introduced with each commit?",
     "evidence": ["-p show the patch introduced"], "answer": ["-p"]},
    {"q": "What is the git log -S option colloquially called?",
     "evidence": ["colloquially referred to as git's \"pickaxe\""], "answer": ["pickaxe"]},
    {"q": "How do you prevent merge commits from cluttering up the git log history?",
     "evidence": ["simply add the log option --no-merges"], "answer": ["--no-merges"]},
    {"q": "How do you redo your last commit after forgetting to add some files?",
     "evidence": ["commit again using the --amend option"], "answer": ["--amend"]},
    {"q": "Which Git version introduced the git restore command?",
     "evidence": ["git version 2.23.0 introduced"], "answer": ["2.23"]},
    {"q": "What name does git clone give to the remote repository automatically?",
     "evidence": ["automatically adds that remote repository under the name"], "answer": ["origin"]},
    {"q": "Why is git checkout -- <file> considered risky?",
     "evidence": ["is a dangerous command"], "answer": ["dangerous", "gone", "lose", "lost", "overwrit", "replace"]},
    {"q": "Who is the author of the commits shown in the git log example?",
     "evidence": ["author: scott chacon"], "answer": ["scott chacon"]},
    {"q": "What does the --relative-date option of git log do?",
     "evidence": ["--relative-date display the date in a relative format"], "answer": ["relative", "2 weeks ago"]},
]

_TR = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "—": "-", "–": "-"})


def norm(t):
    return re.sub(r"\s+", " ", t.translate(_TR)).lower().strip()


def is_hit(chunk_text, item):
    t = norm(chunk_text)
    return any(norm(e) in t for e in item["evidence"])


def is_correct(answer, item):
    t = norm(answer)
    return any(norm(a) in t for a in item["answer"])
