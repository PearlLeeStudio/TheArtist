# TheArtist Chords — Logic Pro Scripter plugin

A DAW-side client of the shared music service: the editor's arrangement
(voicings, comping rhythm, swing, bass line) exported as a
[Scripter](https://support.apple.com/guide/logicpro/use-the-scripter-plug-in-lgcecc16550/mac)
MIDI FX script that plays inside Logic Pro and follows Logic's transport.
It uses the MIDI exporter's shared harmony/bass note placement
(`backend/app/services/smf_export.py`). The script omits drums, and Logic's
instrument and controller handling determine the sound. Note parity is tested;
audible equivalence and transport behavior in the real host remain unverified.

## The song form is the frame

TheArtist works on a song of a set length: a form of sections, each with its
bars. A DAW is the opposite, an open timeline with no end. The plugin keeps
TheArtist's frame instead of trying to fill Logic's: it plays **the whole form
or one chosen section**, starting at the bar you pick, and stays silent beyond
it. It never invents bars past the form. To lengthen or reshape the song,
change the form in TheArtist (Song Form, Complete the form) and export again.

## Use it

1. In the editor, **Export ▸ Logic Scripter script** (or `POST /api/export/scripter`
   with an `ExportSpec`). You get `<title>.theartist.js`.
2. In Logic Pro, on a software-instrument track: **MIDI FX ▸ Scripter ▸ Open Script
   in Editor**, replace the editor's text with the file, press **Run Script**, then
   close the editor. Save it as a Scripter preset if you want to reuse it.
3. Pick what to play in **Section** — *Whole form* or one section such as
   *Chorus (bars 9-16)* — and the Logic bar where it starts in **Start bar**.
   Press Play. Locating and cycling retrigger correctly; notes you play on the
   track pass through.

Parameters: **Section**, **Start bar**, **Transpose** (±12), **Harmony** /
**Bass** on-off, **Velocity %**, **Loop** (repeats the chosen section or the whole
form), and **Beat unit** (Logic's ruler counts the meter's denominator as one
beat; switch to *Quarter note* only if a non-4/4 song drifts in your Logic
version). A note that would ring past the chosen section is released at its edge.

## What it is not

- No network: the script carries its form and notes. Regenerate the file after
  editing the score.
- No drums: Logic's own drummer or the exported `.mid` cover that; the script
  plays harmony and bass.
- No native AU/VST3 yet. That plugin (JUCE, host transport and project
  insertion) is a separate build that needs CMake and a JUCE checkout; this
  Scripter client needs nothing but Logic.

## Tests

`plugin/logic-scripter/test/scripter-host.mjs` is a stand-in for the documented
Scripter API (timing info, parameters, NoteOn/NoteOff, Trace). The test drives
the script through play, stop, locate, cycle wrap, the silent timeline after the
form, loop, section choice and section loop, transpose, 6/8, pass-through and the
committed example:

```
node --test /absolute/path/TheArtist/plugin/logic-scripter/test/theartist_chords.test.mjs
```

Replace `/absolute/path/TheArtist` with your checkout path. The mock is not Logic; the real
check is step 3 above.
