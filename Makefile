# Cross-platform: Windows (cmd.exe) and POSIX. The layout of a venv differs by
# platform (Scripts\python.exe vs bin/python), and cmd.exe has none of `test`,
# `ls`, `find` or `rm`, so every recipe below is either pure make or a python
# one-liner that runs identically on both.
VENV := .venv

ifeq ($(OS),Windows_NT)
  # Pin the shell: ezwinports/mingw make silently switches to sh.exe if one
  # happens to be on PATH, and then these cmd recipes break for that student only.
  SHELL := cmd.exe
  .SHELLFLAGS := /C
  PY := py -3.12
  PYSYS := py -3
  # Backslashes, not slashes: cmd.exe refuses to execute `.venv/Scripts/python`.
  PYBIN := $(VENV)\Scripts\python.exe
  RM_PYTEST_CACHE := if exist .pytest_cache rmdir /s /q .pytest_cache
else
  PY := python3.12
  PYSYS := python3
  PYBIN := $(VENV)/bin/python
  RM_PYTEST_CACHE := rm -rf .pytest_cache
endif

BOT ?= rookie
# `AS` is a GNU make BUILT-IN (the assembler, default `as`), so `AS ?= all`
# never fired and a plain `make spar BOT=rookie` ran `spar.py --as as`, which
# argparse rejects. `?=` only assigns when a variable is UNDEFINED, and make had
# already defined this one. Keep the documented `AS=defender` interface working
# by honouring AS only when it really came from the command line.
ROLE ?= all
ifeq ($(origin AS),command line)
ROLE := $(AS)
endif

.PHONY: install spar ui validate validate-bots qualify submit test clean \
        check-no-key check-referee check-world doctor

# Comments must sit at column 0, NOT inside the recipe: a tab-indented `#` line is
# handed to the shell, and cmd.exe has no comment syntax ("'#' is not recognized").
#
# --seed is REQUIRED: `uv venv` alone creates a venv with no pip, so the pip line
# below died with "No module named pip" on a fresh clone. The stdlib fallback
# seeds pip on its own. --clear because current uv ERRORS OUT on an existing
# .venv ("A virtual environment already exists") instead of replacing it, so a
# second `make install` failed on every machine that had already run the first.
# --no-compile: pip's byte-compile step dies with a bare `assert os.path.exists(
# pyc_path)` AssertionError on this Windows checkout (deep venv path under a
# directory with an `@` in it). The .pyc files are a startup-time optimisation
# only - python regenerates them on first import - so skipping them costs nothing.
install:
	uv venv --python 3.12 --seed --clear $(VENV) || $(PY) -m venv $(VENV)
	$(PYBIN) -m pip install -q --no-compile --upgrade pip
	$(PYBIN) -m pip install -q --no-compile pytest
	@echo ready. no api key needed, ever.

spar:
	$(PYBIN) spar.py --bot $(BOT) --as $(ROLE)

ui:
	$(PYBIN) -m kit.arena_ui.build_ui
	$(PYBIN) -m kit.arena_ui.serve --open

# Always validate against the REAL exported world. Without --world the validator falls
# back to kit/world/fixture.py's ~40-page synthetic world, where every real anchor fails
# to resolve — 15 spurious failures that look like a broken deck and are not.
WORLD := $(firstword $(wildcard kit/world/*/manifest.json))
# Fires only when a recipe that uses it actually runs (recipes expand at run time),
# which is what the old `@test -n ...` guard did with a shell that Windows lacks.
WORLD_DIR = $(if $(WORLD),$(dir $(WORLD)),$(error no world exported - run 'make check-world'))

validate:
	$(PYBIN) validate_deck.py deck/deck.json deck/lineup.json --world $(WORLD_DIR)

# Unrolled: the old `for b in ...; do ... done` was a POSIX shell loop.
validate-bots:
	$(PYBIN) validate_deck.py bots/rookie/deck.json    bots/rookie/lineup.json    --world $(WORLD_DIR)
	$(PYBIN) validate_deck.py bots/operator/deck.json  bots/operator/lineup.json  --world $(WORLD_DIR)
	$(PYBIN) validate_deck.py bots/adversary/deck.json bots/adversary/lineup.json --world $(WORLD_DIR)

# `qualify` used to run a `qualify.py` that was never written, writing a
# `submissions/radar.json` that NOTHING in either repo reads. It is not a
# missing dependency, it is a promise that was never wired up. The student's
# real conformance check is the public suite: `make test`.
qualify:
	@echo make qualify: retired - nothing consumed submissions/radar.json.
	@echo Your conformance check is 'make test' (the public suite).
	@echo Then: make validate ^&^& make submit TEAM=your-team
	@exit 1

# NOT `validate qualify` — qualify is retired (above), and kit.submit REQUIRES
# --team, which this target never passed, so `make submit` failed twice over.
submit: validate
	$(if $(TEAM),,$(error usage: make submit TEAM=<your-team-name>))
	$(PYBIN) -m kit.submit --team $(TEAM)

test: check-no-key
	$(PYBIN) -m pytest tests/

# The referee in kit/ is a hash-synced copy of the arena's (CONTRACTS.md 2.4): students
# must be able to run the exact verifier that will judge them, or prosecution is guesswork.
check-referee:
	@$(PYBIN) -c "import pathlib,sys; sys.exit('kit/referee missing - ask your instructor to run tools.sync_referee') if not pathlib.Path('kit/referee').is_dir() else None; from kit.referee.rubric import CLASSES; from kit.referee.adjudicate import LOCAL_ONLY; print('referee:', len(CLASSES), 'classes, local_only=', LOCAL_ONLY)"

# The world artifact is exported by the instructor; without it nothing can run.
# One python call does all three checks the shell used to do with ls/&&/!.
# counts carries its own `__total__` key, so the original `sum(counts.values())`
# double-counted and printed 24750 for the 12375-page world the README documents.
check-world:
	@$(PYBIN) -c "import json,glob,sys; ms=sorted(glob.glob('kit/world/*/manifest.json')); sys.exit('no world in kit/world/ - ask your instructor for the world artifact') if not ms else None; sys.exit('FAIL: truth.json must never ship to students') if glob.glob('kit/world/*/truth.json') else None; m=json.load(open(ms[-1])); c=m.get('counts',{}); print('world', m.get('world_id'), '-', c.get('__total__', sum(v for k,v in c.items() if not k.startswith('__'))), 'pages')"

doctor: check-no-key check-world check-referee validate
	@echo ready to spar.

# A shipped gate, not a formality: the student kit must contain no model client and no
# API key. It is a real module with its own tests, not a grep — the grep version fired on
# the sandbox's own network-denial probe and on the injection fixtures that have to NAME
# the key to be realistic. Naming a secret is not leaking one; see kit/gate_no_key.py.
check-no-key:
	@$(PYBIN) -m kit.gate_no_key

clean:
	@$(PYSYS) -c "import shutil,pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	@$(RM_PYTEST_CACHE)
