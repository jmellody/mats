"""Patch wrappers into data/pairs_raw.json without another API call.

    python add_wrappers.py

Wrappers are the neutral conversation openers. They carry no expertise signal
by design -- they exist only to give the pair sentences somewhere to sit and to
vary surface context across samples. Nothing about your result depends on them
being model-generated, so hand-written is fine and arguably safer: these were
checked for jargon and for confusion cues, which is the only property that
matters.
"""

import json

WRAPPERS = [
    "I've got responses from a customer survey we ran last quarter and I'm trying to work out whether the difference between two regions is worth writing up.",
    "We collected readings from about sixty sensors across the plant over three months and I need to figure out what to do with them.",
    "I have enrolment and outcome data for a tutoring programme at four schools and I'm trying to work out whether it did anything.",
    "My team pulled sales figures for the last two years by store and I'm trying to make sense of the pattern before the review meeting.",
    "I've got yield measurements from a field trial with three different treatments and I'm not sure how to summarise what happened.",
    "We ran a small pilot with about ninety participants and now I need to decide what we can reasonably say about the results.",
    "I have appointment and no-show records for a clinic going back eighteen months and I want to understand what drives the cancellations.",
    "I'm looking at engagement numbers for two versions of an onboarding flow and trying to decide which one we should keep.",
    "I've got water quality samples from twelve sites collected monthly and I'm trying to work out whether anything is actually changing.",
    "We have shift-level productivity data from the warehouse and I need to figure out what's behind the variation between teams.",
    "I've got test scores before and after a curriculum change across a few dozen classrooms and I'm trying to interpret them.",
    "I have delivery times for about four thousand orders and I'm trying to work out where the delays are coming from.",
    "We surveyed our members about renewal intentions and I'm trying to figure out what the responses actually tell us.",
    "I've got patient recovery times from two different treatment protocols and I need to summarise them for a report.",
    "I have attendance and performance records for a training programme and I'm trying to work out whether they're related.",
    "We tracked app usage for a few thousand people over six weeks and now I'm trying to make sense of the drop-off.",
    "I've got soil measurements from plots under two different management approaches and I'm trying to interpret the differences.",
    "I have match statistics for a season across every team and I'm trying to work out what actually predicts the results.",
    "We collected feedback scores from customers after support calls and I'm trying to understand what's driving the low ones.",
    "I've got equipment failure records going back four years and I'm trying to work out whether the new maintenance schedule helped.",
    "I have income and spending data for a few hundred households and I'm trying to summarise the patterns sensibly.",
    "We ran an experiment with two versions of a landing page and I'm trying to decide what the numbers actually support.",
    "I've got germination rates from several seed batches under different conditions and I need to work out what to conclude.",
    "I have wait times recorded at three different service desks and I'm trying to figure out whether the differences matter.",
    "We collected air temperature and humidity readings across a building for a month and I'm trying to interpret the variation.",
]

if __name__ == "__main__":
    path = "data/pairs_raw.json"
    d = json.load(open(path))
    d["wrappers"] = WRAPPERS
    json.dump(d, open(path, "w"), indent=2)
    print(f"{len(d['pairs'])} pairs, {len(d['wrappers'])} wrappers -> {path}")
    print("Next: python make_stimuli.py filter")
