# TheArtist backend

FastAPI services for chord generation, harmonic analysis, score import,
reharmonization, arrangement, and editable MIDI export. Install from the complete
repository because inference imports the sibling `ai/` tree.

See [setup and serving](../README.md#getting-started), the [API reference](../API_SPEC.md),
and [environment settings](../deploy/env.example). The app reads process
variables and does not load a `.env` file automatically.

The current editor uses local analysis and the Music Transformer. Every model call
goes to a TheArtist Music Transformer checkpoint, and no model-provider API key is
read. Prepare Open Music Theory using `scripts/setup_omt.py`; no book content or
private index is bundled. Berklee retrieval remains off.

Generation can return explicitly labeled rule-based results if a checkpoint is
missing. Reharmonization and model-based review require the selected checkpoint.
Inspect response source/model fields to verify which engine actually ran.

The optional MCP server is in `app/mcp_server.py`, and `theartist_client/` contains
the Python client. Public-host mode disables MCP and local desktop operations.
