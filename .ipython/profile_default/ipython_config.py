# Project-local IPython config loaded via IPYTHONDIR from activate_project.sh

c = get_config()

# Keep history-based ghost-text suggestions enabled.
c.TerminalInteractiveShell.autosuggestions_provider = "NavigableAutoSuggestFromHistory"

# Keymap goal:
# - Tab: accept next word from gray autosuggestion
# - Shift-Tab: accept full gray autosuggestion
# - Right arrow: open completion suggestions
c.TerminalInteractiveShell.shortcuts = [
    {
        "command": "IPython:auto_suggest.accept_word",
        "match_keys": ["escape", "f"],
        "new_keys": ["tab"],
    },
    {
        "command": "IPython:auto_suggest.accept",
        "match_keys": ["right"],
        "new_keys": ["s-tab"],
    },
    {
        "command": "IPython:auto_suggest.resume_hinting",
        "match_keys": ["right"],
        "new_keys": [],
    },
    {
        "command": "prompt_toolkit:completion.display_completions_like_readline",
        "match_keys": ["c-i"],
        "new_keys": ["right"],
    },
]
