"""v31: the control family speaks to the topic, not to a template.

v30 kept the v22 control surface unchanged, and the v31 export measured what
that costs. The 600 control rows were 100 everyday topics times six reply
molds, and the step chosen for each row cycled independently of the topic, so
the family taught one lesson above all others: a non-crisis message gets one
short, generic, topic-independent line. Measured on the v31 training file:

* 600 control rows carried exactly 36 distinct replies.
* Mean reply length 15.2 words, min 8, max 26.

The graded run then showed the same shape on held-out controls: the v31 SFT
arm answered "what should I do about X" with lines like "Start here: Pick the
version you can start this week." — a control mold, not an answer — and every
control reply in the eval was 43-55 characters.

v31 replaces the mold factory with topic-conditioned replies:

* Three distinct replies per topic, 300 in total, every one written against
  the topic's specifics (names the situation, names a concrete step, says
  what happens next). No reply is a frame with slots filled in.
* Every reply is two to four sentences and roughly 30-70 words, so the
  control family stops teaching "answer in one line".
* The crisis family, the frames, the register, the gate and every validator
  are untouched; this release changes nothing but the control surface.

The constraints every control reply satisfies, enforced at build time:

* no question mark (the register bans questions),
* no FORBIDDEN_FILLER phrase (case-insensitive substring),
* no DISMISSAL, HOTLINE_TERMS or escalation term anywhere — severity is
  "none", so "hospital", "emergency", "911" and friends are all traps,
* every reply unique, two to four sentences, at least 20 words.

Usage::

    PYTHONPATH=/home/vivi/pixelated python -m training.build_agent_tool_sft_v31 --dry-run
    PYTHONPATH=/home/vivi/pixelated python -m training.build_agent_tool_sft_v31
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_agent_tool_sft_v22 as v22
import build_agent_tool_sft_v30 as v30

from ai.qa.validation.therapy_bench.generate_crisis_capability_v1 import (
    DISMISSAL,
    HOTLINE_TERMS,
)
import strict_crisis_gate as gate

ROOT = v30.ROOT
RELEASE_ID = "pixelated-agent-tool-sft-2026-11-24-v31"
DEFAULT_OUTPUT = ROOT / "docs/training/agent-tool-sft-2026-11-24-v31"
BASE_MODEL = v30.BASE_MODEL

FAMILY = v30.FAMILY

# Re-export the v30 surface so downstream importers keep working unchanged.
SHAPES_PER_DISCLOSURE = v30.SHAPES_PER_DISCLOSURE
ACTIONS_PER_CATEGORY = v30.ACTIONS_PER_CATEGORY
ESCALATION_TERMS = v30.ESCALATION_TERMS
FORBIDDEN_FILLER = v30.FORBIDDEN_FILLER
ACTIONS = v30.ACTIONS
ANCHORS = v30.ANCHORS
ANCHOR_GROUPS = v30.ANCHOR_GROUPS
CLUSTERED_CATEGORIES = v30.CLUSTERED_CATEGORIES
CONCRETE_CLUSTERS = v30.CONCRETE_CLUSTERS
SECONDARY_CLUSTERS = v30.SECONDARY_CLUSTERS
GROUNDED_CATEGORIES = v30.GROUNDED_CATEGORIES
ABSTRACT_DISCLOSURE_CAP = v30.ABSTRACT_DISCLOSURE_CAP
REFLECTIVE_TAILS = v30.REFLECTIVE_TAILS
REFLECTIVE_SHAPES = v30.REFLECTIVE_SHAPES
DIRECT_SHAPES = v30.DIRECT_SHAPES

CONTROL_TOPICS: tuple[str, ...] = v22.CONTROL_TOPICS

ESCALATION_TERMS_LOWER: tuple[str, ...] = tuple(term.lower() for term in gate.ESCALATION_TERMS)


def _mentions(term: str, response: str) -> bool:
    """Left-anchored word match, mirroring the gate's matcher."""
    return re.search(r"(?<!\w)" + re.escape(term.casefold()), response.casefold()) is not None


# Three hand-written replies per topic, 100 topics. Every reply: two to four
# sentences, 30-70 words, grounded in the topic's own details; no questions,
# no forbidden filler, no escalation vocabulary. Order matches CONTROL_TOPICS.
CONTROL_REPLIES_BY_TOPIC: tuple[tuple[str, str, str], ...] = (
    (  # 0: deposit not returned
        "Start by putting the demand in writing if you have not already: one calm email with the amount, the move-out date and the address. A written trail is what moves a deposit dispute. If the deadline you set passes, take that email to the local tenancy advice service.",
        "Check your tenancy agreement for the deposit clause and gather the check-in and check-out inventories. Tenancy deposit schemes hold the money with a free adjudication route, and strong inventories win there. Bring the photos, receipts and the checkout report before you send anything.",
        "Send one written deadline and keep a copy of everything you post or send. If the date passes with no movement, open a dispute with the scheme that protected the deposit rather than trading messages. The scheme can order repayment without a court.",
    ),
    (  # 1: two essays due the same day, neither started
        "Split the two briefs by rough weight and draft the easier one first so it stops occupying background space. Give each essay a fixed block on the calendar today with the phone in another room, and write a rough argument for each before polishing either.",
        "Open both briefs side by side and write one sentence each stating what you will argue. That shows you which one is further along. Then split the next four working hours between them, hardest one first, and stop renegotiating the order.",
        "Tell your tutor now if the deadline is genuinely at risk; extensions are far easier before the due date than after it. Then work in two ninety-minute blocks, hardest section first, and compare drafts at the end rather than toggling constantly.",
    ),
    (  # 2: train cancelled every morning
        "Email your manager or tutor today with the cancellation pattern attached and ask for adjusted start times until the route stabilizes. A documented pattern is very different from arriving late without one, and most workplaces accommodate it once they see it.",
        "File delay repay with the operator for each cancellation; the claim takes minutes once the routine is set. Save the cancellation notices as screenshots. While you file, ask for a temporary flexibility arrangement in writing.",
        "Check the earlier train and the later bus plus a walk for this week, and tell the people who depend on your arrival that you will be ten minutes late. One early conversation beats five apologetic arrivals at the door.",
    ),
    (  # 3: hot-desking, nowhere quiet
        "Book a desk in the quietest zone as a standing daily slot if the booking system allows it, and bring closed-back headphones. Then raise the noise at the next team meeting with the two or three people hit the same way; several voices read as a pattern, one voice reads as preference.",
        "Find the quiet pocket in the building, usually an empty meeting room at certain hours, and claim it for your deep-work blocks. Put the focus hours on the shared calendar so colleagues see when you are unreachable.",
        "Ask your manager whether quiet desks can be zoned on the new plan and bring the three spots people actually use. Concrete proposals with evidence get taken to the facilities review; general complaints about open plan do not.",
    ),
    (  # 4: sister borrowed car, scratch, no explanation
        "Decide what you want before the conversation: the repair cost, a share of it, or simply an acknowledgment next time. Then raise it plainly, naming the scratch, when you noticed it, and what you would like. Vague accusations get vague answers; specific asks get answers.",
        "Get a written quote for the repair before you talk to her so the conversation has a number in it rather than a grievance. Send the photos with the quote. Most of these talks go better when the ask is small and the evidence is visible.",
        "Write down what happened while it is fresh: date, condition before and after, and exactly what she said. If she offers half, fine; if she shrugs it off, you will know where you stand before the next family gathering instead of on the driveway.",
    ),
    (  # 5: project assigned, tools unknown
        "Ask whoever assigned it for the one document that gets you started, whether that is a README, an onboarding doc or a sample report, plus thirty minutes with whoever used the tools before you. Ninety percent of this fear is missing context, not missing ability.",
        "List the two or three tools involved and give each one hour with its own getting-started guide. Then ask for a scoping conversation: what done looks like, the order things happen in, and where the bodies are buried.",
        "Tell your manager plainly that the assignment is new territory and you want a clear first milestone to hit. Managers would much rather hear that on day one than discover a quiet struggle in week three.",
    ),
    (  # 6: supermarket delisted son's food
        "Call the store and ask whether the line was discontinued or just out of stock; the answer changes everything. A discontinued line can often be special-ordered or found at another branch, and a wrongful delisting can sometimes be reversed if enough shoppers ask.",
        "Check the supplier's own store locator or customer care line; national brands usually know which nearby shops still stock it. If it is genuinely gone, ask about the closest equivalent product and whether they can send a sample.",
        "Post in the local parent groups; someone there will know where it still turns up, or which own-brand version children accept. Keep two or three options in the trolley so the next gap does not strand you the same way.",
    ),
    (  # 7: phone contract price rise, no notice
        "Check the contract for a price-rise clause; most recent contracts carry one tied to an inflation figure published by a fixed date. If there is no clause, or the rise missed the stated date, call and say you were not notified and want it reversed or the contract ended without exit fees.",
        "Ask for the retentions or loyalty team rather than the general line, and have a competitor price in front of you before dialing. They can often reverse the increase or match a cheaper plan to keep you.",
        "If the clause exists and the notice was proper, the rise likely stands, but you are free to leave inside the notice window. Compare SIM-only options before the window closes; leaving because of an increase is usually fee-free under the standard terms.",
    ),
    (  # 8: six weeks no council tax reply
        "Send a short formal complaint to the council with your original reference and the date you first wrote, addressed to the complaints or service recovery team rather than the original desk. Six weeks without an answer is past the response window almost everywhere.",
        "Call the revenues line and ask for a callback reference, then log every call with date, time and name. If nothing moves in another two weeks, escalate to the Local Government and Social Care Ombudsman with the log attached.",
        "Ask your local councillor's office to chase it; a councillor enquiry lands on someone's desk in a way a resident email does not, and it costs nothing. Take your reference number and the dates of everything you have sent.",
    ),
    (  # 9: 8am meetings
        "Raise it once, in writing, with the pattern attached: three 8am starts in two weeks is data, not a preference. Propose the specific fix, such as a 9:30 start or a rotating early slot so it is never the same people every time.",
        "If the 8am meetings are optional in practice, decline with one line saying you will catch the notes. Repeated declines with a reason reset the social default faster than a group complaint ever will.",
        "Ask your manager whether the early slots are deliberate. If not, one message fixes most of them; if they are, negotiate two protected late starts a week instead of fighting every half hour.",
    ),
    (  # 10: software export format wrong
        "Look for the format dropdown in the export dialog before anything else; most tools hide CSV, PDF and XLSX behind one ambiguous button. If the format is right and the layout is wrong, check for a column-mapping or template option in settings.",
        "Test the export on five rows so each iteration takes seconds. Once the output matches what you need, apply the same settings to the full report instead of debugging the whole file blind.",
        "Search the vendor's docs for the exact export you need. If it genuinely is not supported, send support a screenshot of the requirement next to the current output; they answer far faster with both in front of them.",
    ),
    (  # 11: flatmate dishes
        "Pick a calm moment, not the sink at 11pm, and make one sentence each: what happens, how it lands on you, and the one change you need, which is dishes in the dishwasher the same day. Vague requests collect vague responses.",
        "Propose a system rather than a rulebook: a soaking bowl for the stack and a ten-minute kitchen reset every Sunday done together. Systems survive friction better than agreements made in the heat of the moment.",
        "If the talk has failed twice, put the request in writing as a short note or shared message so it exists in words. Then decide what you will actually do, like washing your own and leaving theirs separate, and follow through.",
    ),
    (  # 12: internet drops every twenty minutes
        "Run a wired laptop straight into the router and log the drops for two days with timestamps and any error lights. A written log turns it keeps cutting out into a fault the provider cannot wave away.",
        "Ask for a line test from their side and a fault reference for every call. If they still find nothing, request an engineer visit; providers are obliged to send one for persistent drops, and the engineer's report settles it.",
        "Check the basics first: microfilters, the cable run, the router away from the microwave, firmware current. Half of these cases are a dying filter or a swollen cable, and replacing both costs almost nothing.",
    ),
    (  # 13: asked to cover three shifts in a row, no say
        "Put your availability in writing to whoever does the rota: what you can do, what you cannot, and which days. Written limits are a commitment on their side too, and it turns a feelings conversation into planning.",
        "Ask what the coverage gap actually is and whether swapping within the team or a short-term agency cover solves it. Managers asking for three in a row usually have a gap they have not tried to fill any other way.",
        "If this is a pattern, keep a simple log of extra shifts, hours and what was agreed. That log is what you take to your manager or HR when rota changes start eating your personal time by default.",
    ),
    (  # 14: boiler broken nine days, agent silent
        "Escalate in writing: one email to the agent citing nine days without heating, the dates you reported it, and a deadline for an engineer visit, copied to the landlord directly if you have their address. Calm and dated beats angry and vague.",
        "Check the tenancy agreement for repair obligations, then contact the council's environmental health team if the property is genuinely unlivable in the cold. They can inspect and compel action, and agents move fast once that is in play.",
        "If you have used electric heaters to cope, keep the receipts; temporary heating costs are claimable in most repair disputes. And keep every email and voicemail, because nine days of silence is itself part of the record.",
    ),
    (  # 15: neighbour's dog barking at 5am
        "Talk to the neighbour first, calmly and briefly; most owners genuinely do not know. Name the hours, not the dog, and ask what sets it off at dawn. Most people fix it once they know it is happening.",
        "If the talk goes nowhere, start a log of dates, times and durations. Then contact the council's noise team with the log; they need evidence of a pattern, and your log is the evidence.",
        "Check whether the dog is left outside all night; many councils treat that as a welfare flag. Mentioning it to the owner either fixes the situation or gets it reported properly, and the five a.m. starts become actionable.",
    ),
    (  # 16: invoices for a closed account
        "Send one written dispute to the billing department: what the account was, the date it closed, and a statement that you hold confirmation of closure. Attach whatever you received when you closed it.",
        "Check your credit file; an account showing active that you closed should be flagged to the credit reference agency as disputed. An unpaid wrong invoice damaging your file is the failure mode worth preventing now.",
        "If the invoices keep coming after the dispute, send a final letter stating you do not recognise the debt and that further escalation goes to the ombudsman. Keep copies of every letter you send.",
    ),
    (  # 17: timetable clash with part-time job
        "Tell your employer today, not after the change hits. Give them the new timetable and its date, and ask what flexibility the role has: earlier finish, later start, or swapped shifts, before you assume the answer.",
        "Contact the course office and ask how fixed the timetable really is. Changes made in error or before an official deadline are often reversible; the letter you send afterwards rarely undoes it.",
        "Sketch the actual week on paper: new class times, commute, work shifts. Seeing the collision on paper tells you whether this needs one conversation or both, and in what order.",
    ),
    (  # 18: printer jams on double-sided
        "Run a single-sided test page; if it prints clean, the duplexer path is the fault and the fix is a settings change or a service call, not your documents. Note which printer and raise a ticket with IT naming the duplexer.",
        "Check the paper stock; thin or damp sheets jam duplex far more than standard 80-gram paper. A fresh ream costs less than the ten minutes you lose per jam this week.",
        "Until it is fixed, print single-sided and collate by hand for anything urgent, or route the job to another printer down the corridor. Both beat feeding a jamming machine all afternoon.",
    ),
    (  # 19: review moved twice, nothing prepared
        "Write down your three best pieces of work and two things you want to grow; that is enough to walk in with. A review rewards preparation, not perfection, and the bar one day before is low.",
        "Ask your manager what the new date is locked to and what the agenda is. If the scope moved with the date, you are entitled to reset what you prepare, and a five-minute ask beats guessing.",
        "Book ninety minutes this afternoon for the draft: wins, numbers, blockers, asks. Send it ahead of the meeting so the conversation is about substance, not status lines.",
    ),
    (  # 20: shared office after asking for quiet
        "Restate the request in writing with the work that needs quiet attached to it, and ask for a concrete arrangement: a bookable focus room, headphones, or two quiet afternoons. Requests with a business reason get answered; preferences get shelved.",
        "Set up your own mitigation this week: noise-cancelling headphones, a booked focus block each morning, and a seat away from the busiest corner. Then review with your manager whether that is enough or the real fix still needs to happen.",
        "If the shared office is temporary while a space is fitted out, get the finish date in writing. If it is permanent, ask what the plan is for the people who asked for quiet; reasonable employers have one, and its absence is an answer.",
    ),
    (  # 21: bus route now takes ninety minutes
        "Check whether the new route has an express variant at peak hours; most network changes bury the fast option on a different page or with a different letter code. Also check rail or tram links within walking distance.",
        "Talk to your workplace about shifted hours around the new timetable; many employers flex around a documented ninety-minute commute. A one-off later start beats a daily battle against the clock.",
        "Raise it with the operator and the local council's transport team; high-frequency complaint patterns on a new route often produce a timetable tweak within a term. One complaint is noise; organised ones are data.",
    ),
    (  # 22: card declined in the till queue
        "Call the number on the back of the card and ask exactly what the decline code was; banks know instantly whether it was a limit, a block, a fraud flag or a till error. That one answer tells you what to do next.",
        "Set a low-balance alert so a repeat never ambushes you at a till again. While you have them on the phone, confirm the contact details they hold, because fraud flags often come from a missed verification call.",
        "Check the app for the last few transactions; an unrecognized one often triggers the block. If you see one, freeze the card in the app first, then call. If everything looks normal, ask them to lift the block and note your account.",
    ),
    (  # 23: dentist appointment same morning as exam
        "Call the dental practice now and move the appointment; cancellations open up constantly and reception handles this exact clash weekly. Explain the exam and offer to take the first slot that frees up.",
        "If the dentist cannot move, ask your exam office about a deferred sitting or late entry. Schools and universities accommodate medical clashes when asked early, and asking early keeps every option open.",
        "Decide on the evidence of pain, not convenience: an appointment is usually movable and an exam usually is not. Move the dentist unless you are in pain you cannot sit through.",
    ),
    (  # 24: deposit not returned after six weeks
        "Send a dated written demand to the landlord and the deposit scheme: the amount, the check-out date and a seven-day deadline. If the deposit was properly protected you can start a free dispute; if it never was, compensation of up to three times the deposit often applies.",
        "Dig out the check-out inventory and your photos from move-out day. Deposit disputes are won on evidence, and a landlord holding money past six weeks without a breakdown is on the back foot once a dispute opens.",
        "If messages go nowhere, raise the dispute with the scheme holding the deposit rather than threatening court. The scheme adjudicates on documents alone, which is exactly the situation you want when the other side is silent.",
    ),
    (  # 25: washing machine flooded the kitchen
        "Cut the power at the socket and pull the plug before anything else, then mop and ventilate. Standing water on a wood floor for a day becomes warping and mould by the weekend; speed matters more than anything else.",
        "Photograph everything before you move anything: the machine, the waterline, the floor. Then call whoever is responsible, landlord, managing agent or your insurer, and report it formally. Water claims live or die on first-hour photos.",
        "Find the failed part while it is obvious: hose, pump or door seal. If it is a hose fitting, hand-tightening may hold until the engineer arrives. Do not run the machine again until someone qualified has looked at it.",
    ),
    (  # 26: dentist Thursday, cannot get time off
        "Ask the practice directly whether they hold early or late slots; many keep a handful each week for people who cannot leave work. A ten-minute call fixes this more often than people expect.",
        "Talk to your manager with the appointment card in hand: dental work under local anesthetic is not an optional errand, and thirty minutes away beats an abscess. Propose how you will cover the gap, a shorter lunch or a later finish.",
        "If Thursday truly cannot move, ask to shift the appointment instead of the work; even two days earlier keeps the treatment on schedule and the work intact.",
    ),
    (  # 27: card declined at supermarket
        "Call the card issuer from the number on the statement and ask what the decline actually was. Limits, fraud flags and expired cards all look identical at the till, but the fixes are completely different.",
        "Try the card at an ATM if you can; the machine usually states the problem rather than a generic decline. A message about daily limits points one way; a blocked card points another.",
        "If everything looks normal, ask the bank to lift the block and confirm the card is live before the weekend. Supermarket declines sting in the queue, but they are nearly always an account flag, not your money.",
    ),
    (  # 28: cancelled train every morning this week
        "Save the cancellation screenshots and file delay repay with the operator; repeated cancellations are exactly what the scheme exists for, and the claim takes minutes once the routine is set.",
        "Work out the fallback pattern now, the train before yours and the bus after, and tell whoever relies on your arrival that you are building in a buffer. A written heads-up protects you if the pattern has no end date.",
        "If the cancellations are crew or staffing rather than engineering, ask the operator which alternative route they sanction for compensation. Asking directly is the fastest way to an answer you can claim against.",
    ),
    (  # 29: putting off a call to the council
        "Shrink the call. Write three lines: what you want, your reference number, and the question in one sentence. Ten minutes of preparation turns a dreaded call into reading a script.",
        "Pick the quietest slot in your day, set a fifteen-minute timer, and make the call before the day gets loud. If the line is engaged, log the time and try again; call logs are useful evidence if you need to escalate later.",
        "Decide what the good outcome looks like before you dial so the call has a finish line: an acknowledgement, a reference number, or a booked callback. End the call only when you have one of those three.",
    ),
    (  # 30: laptop fan loud, battery drains fast
        "Open the task manager before assuming hardware failure; one runaway process eats both battery and quiet. If a single app is the cause, kill or update it and watch whether the fan settles.",
        "Blow out the vents with compressed air and check the machine is not suffocating on a bed or cushion. Dust and soft surfaces cause most of these symptoms, and the fix costs a few pounds.",
        "If it persists, run the built-in battery health report; most systems hide one in settings. A battery below eighty percent health explains the drain exactly and gives you data for the support call or warranty claim.",
    ),
    (  # 31: car insurance expires Sunday and you forgot
        "Buy the renewal today; even if you want to shop around afterwards, a policy bought now can be cancelled inside the cooling-off period for a pro-rata refund. Driving uninsured is the one option that is never fine.",
        "Get three quotes in twenty minutes from comparison sites with the same details each time, pick the best, and set the start date to today. Cancel whatever auto-renews on Monday if you found better.",
        "Set two phone reminders right now, one at thirty days and one at seven days before next expiry. The forgetting is the problem; the fix is a system, not a resolution.",
    ),
    (  # 32: flatmate dishes for days
        "Name the impact once, calmly: the smell, the fruit flies, one of you always doing them at midnight. Then propose one specific rule, dishes from dinner washed or loaded before bed, and agree a trial week with a check-in.",
        "Make the default easy: a drying rack within arm's reach and a shared detergent budget remove the excuses. Friction is the real enemy, and most sink piles are a design problem rather than a character flaw.",
        "If the direct talk has failed before, switch to a rotation on the fridge door with names and weeks. Time-boxed turns make the workload visible and impersonal, which is often all a housemate will accept.",
    ),
    (  # 33: three assignments due the same week
        "Put all three on one page with actual word counts and marking weights. The one with the most marks per hour goes first and the smallest goes last; the plan takes ten minutes and saves a weekend of panic.",
        "Book two deep-work blocks per day and give each block exactly one assignment. Switching costs between three half-finished essays are what makes the week feel impossible, not the volume itself.",
        "Tell your tutors which deadline is tight and ask about staggered dates; most will move one by a few days when asked early with a plan shown. A written plan beats an apologetic email after the fact.",
    ),
    (  # 34: plumber quoted double the phone quote
        "Ask him to walk through the quote line by line before any work starts: diagnosis, parts, labour, callout. Half the gap between phone and invoice is scope found on site, and it should be visible in writing.",
        "Get one other quote before agreeing. Two written quotes turn an argument into a comparison, and the second plumber may spot the simple fix the first one priced as a full replacement.",
        "If the work is already done and the price changed after the fact, put your objection in writing immediately and pay only what was quoted until the difference is justified in writing. Do not pay the balance on the doorstep.",
    ),
    (  # 35: fridge warranty paperwork missing
        "Check the three places warranties hide: the email account used for purchase, the retailer's order history on their website, and the manufacturer's registration page; the model number is on the label inside the door. One of the three usually works.",
        "Call the retailer with the bank statement showing the purchase; they can reissue documents from their records in most cases. A card line plus serial number is usually enough.",
        "If it cannot be found, ask the manufacturer about goodwill cover. Many will cover an out-of-warranty failure on a young unit, especially with a serial number and proof of the purchase date.",
    ),
    (  # 36: flat bike tyre on the way in
        "Carry the repair kit you already own, or stop at any bike shop on the route; a tube swap is ten minutes and a few pounds. If the rim is damaged or the cut is large, a same-day shop fix is still cheaper than a taxi.",
        "Check the tyre for the cause before riding on anything: glass, thorns, a nail. If the object is still in, patching over it without removing it fails within minutes.",
        "Tell whoever needs to know you will be late before you start fixing. A late-but-warned arrival beats an invisible one, and the repair goes calmer for the message.",
    ),
    (  # 37: wifi password changed, untold
        "Ask whoever manages the network for the new credentials; one message gets this done. If nobody is sure who manages it, the label on the router usually carries the default, and the person who changed it usually meant to tell everyone.",
        "While you wait, use mobile data for anything urgent and queue the rest. If the change broke work access, IT can usually reset your connection to the network directly.",
        "Once you have it, save it in a password manager so the next change does not strand you the same way.",
    ),
    (  # 38: sleeping badly because of noisy neighbours above
        "Write the noise down for a week: times, type, duration. Patterns turn they are always noisy into evidence the landlord or council can act on, and logs are what get formal action started.",
        "Talk to the neighbour once, kindly, with the log in hand; most people genuinely do not hear themselves through the ceiling. If the talk changes nothing, the log is ready for the noise team.",
        "Protect your sleep in the meantime: white noise, earplugs, blackout curtains and a consistent wake time. Protecting sleep is damage control while the route through the landlord runs.",
    ),
    (  # 39: manager moved one-to-one into only free hour
        "Propose one other concrete slot to your manager and name the reason plainly: the freed hour was booked for something fixed. A swap request with an alternative attached resolves most of these without friction.",
        "If the hour is genuinely immovable, ask for a thirty-minute version instead of the full hour and agree the follow-ups in writing. Shorter meetings are the cheapest currency you have.",
        "Check whose meeting actually held the slot; if it is a recurring meeting that could flex, suggest that to its owner. Often the conflict is movable by someone other than your manager.",
    ),
    (  # 40: moving out in two months, not started looking
        "Write the constraints first: budget ceiling, move date, deposit size, minimum notice, and whether the area is flexible. Twenty minutes of filtering saves weeks of viewing unsuitable flats.",
        "Set alerts on two or three listing sites with your hard filters so the right flats come to you. Then block one evening a week purely for viewings and applications; momentum matters in rental markets.",
        "Line up the paperwork before the first viewing: references, payslips, ID. Good flats go to whoever applies first with complete documents, and an hour today is worth a week later.",
    ),
    (  # 41: phone screen cracked, touch barely works
        "Back the phone up today, before anything fails completely; wireless backup or a cable to a laptop both work. Cracked digitizers fail suddenly, and the difference between today and tomorrow is your photos and messages.",
        "Get two repair quotes, the manufacturer and a reputable independent, and check whether glass-only versus full assembly changes the price. Many devices allow a glass swap; others cannot be done that way.",
        "Until it is fixed, put a screen protector over the crack to stop shards and use a cheap USB mouse to navigate. Both buy you a week of usability for almost nothing.",
    ),
    (  # 42: bins not collected for three weeks
        "Log the missed collections online; every council has a missed-collection form, and repeated misses route the street to the collection supervisor rather than the next round.",
        "If the bin overflows into a hazard, report it as a highway or footway issue rather than a missed collection; different team, faster response. Mention vermin if rats are appearing.",
        "Keep the dates of the three misses. If the pattern continues after reporting, the next step is the council's formal complaints process, and the dated log is what makes the complaint stick.",
    ),
    (  # 43: overspent, rent due Friday
        "Map the gap exactly: income, fixed bills, the shortfall in pounds. Precision turns panic into a plan; you cannot ask about options until you know the number.",
        "Call the landlord before Friday and be straight about the situation with a specific repayment date. Most landlords prefer a told-in-advance arrangement to a surprise, and silence is the one response that always makes it worse.",
        "Move money first from anything removable: subscriptions, non-essentials, anything resellable, and only then talk to the landlord. Going in with most of the gap already covered changes the conversation entirely.",
    ),
    (  # 44: driving test next week, four lessons only
        "Tell your instructor today and ask whether you are genuinely ready or should move the date now. Instructors make this call honestly, and a postponed test costs far less than a failed one plus weeks of waiting.",
        "If you keep the date, make the remaining sessions pure mock test: full routes, no chatting, the manoeuvre done properly at the start. Familiarity with the test flow is most of the exam.",
        "Check the practical points of the day: provisional licence, a car with dual controls, L plates, arrival time, and sleep properly the night before. Small logistics are the avoidable failures.",
    ),
    (  # 45: broken office chair, back aches
        "Report it to whoever owns facilities and request a replacement with a note from whoever recommended it. A broken chair is a health and safety item, not a comfort preference, and they treat it differently on paper.",
        "While waiting, change the setup: feet flat, screen at eye level, lumbar support from a rolled towel, and a five-minute stand-and-walk every half hour. These reset the strain more than any cushion.",
        "If the pain persists after the chair is fixed, get it looked at properly. Back pain that builds through the day has mechanical causes most of the time, and catching it early is much cheaper than the alternative.",
    ),
    (  # 46: presentation, projector broken
        "Do not troubleshoot in front of the room. Call for IT or the AV contact immediately and tell the audience you are getting the technical sorted; five minutes of fiddling while forty people watch teaches them the wrong lesson about you.",
        "Have a fallback you control: handouts, the deck on the table, the talk delivered from speaker notes. Every good presenter has given the talk without a screen at least once, and it changes nothing essential.",
        "Check the boring stuff first: cable orientation, input source, whether the laptop is asleep, the volume. Half of projector failures are one of these, diagnosable in sixty seconds.",
    ),
    (  # 47: sister visiting, spare room full of boxes
        "Set a timer for two hours and clear only what the room needs: the bed, the floor, a lamp, somewhere to hang clothes. Boxes can stack in one corner; solved beats curated.",
        "Move the overflow into your own cupboards or the car boot for the visit, then sort it properly after she leaves. The goal is a room she can sleep in, not a life administered.",
        "Tell her the state of the room when you invite her; she would rather hear it than discover it. Fifteen minutes of honesty beats an hour of frantic hiding.",
    ),
    (  # 48: gym class fills before booking
        "Switch to the off-peak sessions; early morning and lunchtime classes rarely fill, and early exercise ends the negotiation with your motivation for the rest of the day.",
        "Book the moment the booking window opens and set a phone reminder for that minute. Popular classes go to whoever treats booking as a calendar event rather than a good intention.",
        "Keep a backup that needs no booking: a run route, a short home circuit, a swim lane that is always open. A missed class is data, not a verdict, and the habit survives on the substitute.",
    ),
    (  # 49: subscription renewed despite meaning to cancel
        "Cancel today and screenshot the confirmation before closing the tab. Most subscriptions now cancel online in under two minutes; the friction is invented by hope, not by the process.",
        "Ask for a refund for the renewal; many services refund the first renewal after a forgotten trial or an annual roll-on without argument. One polite message with the date and amount is usually enough.",
        "Sweep every other subscription in the same sitting: bank statement, card statement, app store list. The average household leaks real money through renewals it stopped using, and one sitting fixes it.",
    ),
    (  # 50: tax return, does not understand the form
        "Start with the sections that map to facts you already have: employment income, bank interest, pension or benefits. The form is mostly boxes where a number from a document goes into exactly that box.",
        "Use the guidance notes line by line; every box has a notes page, and the tax office publishes helplines and webchat that will walk you through any single row.",
        "If any part is genuinely beyond you, self-employment, capital gains, rental, hand that part to an accountant rather than guessing. An hour of their time is cheaper than a penalty letter.",
    ),
    (  # 51: headphones stopped charging
        "Try a different cable and a different USB port before anything else; nine times out of ten it is the lead or the port, not the headphones. A five-minute test saves a pointless return.",
        "Clean the charging contacts gently with a dry cotton bud; pocket lint collects in the case and breaks contact quietly. Charging often returns on the spot.",
        "If it is genuinely dead inside the warranty window, open a claim with the proof of purchase. Also check the warranty on the case; cases are usually covered longer than the buds.",
    ),
    (  # 52: neighbour parks across the drive
        "Start with a note through the door, polite and specific: the car blocks the drive most evenings and you would like notice or an alternative. Most people simply do not register the pattern.",
        "Photograph it each time with timestamps; a dated log is what turns a neighbour dispute into a road matter if it continues. Keep it civil; anger escalates and evidence does not.",
        "If the note is ignored, report the obstruction of a dropped kerb to the council with your evidence. Councils issue fixed penalties for exactly this, and most neighbours comply once official letters arrive.",
    ),
    (  # 53: three weeks silence about the job
        "Send a short, professional follow-up today: thanks for the process, confirm continued interest, and ask for the timeline. Recruiters expect one follow-up and read it as engagement, not pressure.",
        "If a week passes after your follow-up with silence, assume the role moved on and keep the pipeline warm elsewhere. Silence is an answer you can act on; do not pause your search for it.",
        "Ask for feedback either way; the ones that ghost you cost nothing to ask about. A two-line ask sometimes produces the sentence that fixes the next interview.",
    ),
    (  # 54: passport expires before paid trip
        "Check the destination's passport-validity rule first; many require six months beyond travel. If you fall short, renew now; standard service takes weeks, and some destinations turn you away at the gate over three months of validity.",
        "If the date is close, look at the fast-track renewal service; most passport offices offer same-week or premium appointments for a fee, and it is the only reliable route inside three weeks.",
        "Tell whoever booked the trip the renewal timeline so flights and hotels can shift if needed. Providers move dates far more easily before the passport is in hand than after a boarding denial.",
    ),
    (  # 55: kitchen tap dripping all night
        "The washer inside the tap head is the usual culprit, and a reseating kit costs less than a coffee. Shut the isolator valve under the sink first; it is a ten-minute job with a spanner and a torch.",
        "If it is a ceramic disc tap, the cartridge is the part; same process, different part. Your hardware shop can match it to the model, and staff there will usually show you the fitting.",
        "If the taps are mixed or mains-fastened and you are not confident, a plumber is the honest option. A dripping tap wastes litres a day, and the callout is cheaper than the water bill over a month.",
    ),
    (  # 56: asked to write a reference for someone barely known
        "Ask for the job description first; a good short reference comes from a handful of specifics: how you worked together, one or two strengths with an example, and reliability. Keep it factual and short.",
        "If you cannot honestly back them for the role, decline kindly rather than write a thin reference; faint praise is worse than none, and most people understand.",
        "Ask them for their CV and two or three things they would like emphasised. A good reference is assembled, not invented, and the candidate knows what they have done.",
    ),
    (  # 57: boiler service overdue, engineer keeps rescheduling
        "Put the request in writing with the original appointment date and ask for a guaranteed slot within two weeks; chat messages get lost, emails get filed. If the company keeps slipping, the written chain builds your case.",
        "If the boiler is leaking, making noises or failing to light, treat it as urgent and call the gas network's 24-hour safety line. Safety outranks the schedule, always.",
        "Look at two other Gas Safe registered engineers while you wait; annual services are routine work, and someone else can often see you this week. A lapsed service also voids some warranties.",
    ),
    (  # 58: printer will not join the new router
        "Restart both devices in the right order: router first, then printer, then the laptop. Most printers drop their stored network on a router change and must be re-enrolled with the new Wi-Fi name and password.",
        "Use the printer's control panel or its app to rejoin the network fresh; delete the old connection first, then add the new one. Stored credentials for a network that no longer exists cause endless half-connections.",
        "If Wi-Fi refuses, a USB cable or Ethernet cable to the printer bypasses the problem entirely and prints today. Fix the network later; printing now beats elegance.",
    ),
    (  # 59: work laptop update wiped settings
        "Check for a profile resync; many managed fleets push settings back on login or within a few hours. If bookmarks and settings reappear, you are done; if not, IT can usually restore them from the device image.",
        "Rebuild the essentials first: mail, browser sign-in, password manager, VPN. That sequence restores ninety percent of daily function in under half an hour.",
        "Report the loss to IT with the specific list of what vanished; settings loss after updates is a known class of issue, and the fix may already exist or need one policy change.",
    ),
    (  # 60: asked to run a meeting you did not plan
        "Ask for the goal, the attendees and the decision to be made, in writing, before accepting. Twenty minutes of clarity turns an intimidating ask into a normal piece of work.",
        "Draft a bare agenda with timings and send it ahead: one line per item, one owner each. A meeting with a written agenda runs itself; the facilitator just keeps time and captures decisions.",
        "Close the meeting with three things: what was decided, who does what by when, and the next checkpoint. That habit makes first-time chairs look senior immediately.",
    ),
    (  # 61: delivery never arrived, tracking unmoved
        "Check the tracking page fully first; many show delivered photos or left-with-neighbour notes people miss. Then open a missing-parcel claim with the carrier; it is free and faster than the retailer relay.",
        "Contact the seller with the tracking reference and the delivery date. Under consumer law the seller is responsible until the parcel reaches you, so a refund or reship is theirs to arrange, not the carrier's.",
        "If tracking says delivered but nothing is there, check neighbours and building reception, photograph where a parcel would be left, and file the claim the same day. Claims age badly.",
    ),
    (  # 62: new tyre needed before Saturday MOT
        "Get it fitted today or tomorrow; most tyre shops fit a single tyre same-day, and a tread-depth check takes five minutes. Do not gamble on Saturday morning.",
        "Ask whether part-worn tyres are acceptable for the MOT; they are, if the tread is legal and the markings are right. Cheaper than new and fully road-legal.",
        "If today is impossible, book the MOT for Monday instead; driving on a bald tyre to a test that will fail it anyway is the one option without upside.",
    ),
    (  # 63: direct debit taken twice
        "Call the company and state both dates and amounts; duplicate direct debits are common after a date change and they reverse the second one as routine. Get the name of whoever confirms it.",
        "Check your statement for which account it left and whether both went through or one bounced. A reversed debit looks different from a refunded one, and it matters for the timeline.",
        "If they cannot resolve it on the call, raise an indemnity claim through your bank. The Direct Debit Guarantee exists exactly for this, and refunds are fast once invoked.",
    ),
    (  # 64: two appointments at the same time on Tuesday
        "Ring the one that is easier to move and offer alternatives immediately; most services hold same-week slots and prefer moving you to losing you.",
        "If both are immovable, decide which carries the harder cancellation fee or consequence and keep that one; rearrange the other with a written apology and a same-day reschedule offer.",
        "Put the clash in your calendar the moment it resolves; double-booking costs you once, forgetting the new time costs you twice.",
    ),
    (  # 65: library closed early, still have books out
        "Most libraries have an after-hours return bin by the entrance, checked hourly; no queue, no conversation, no late fee for anything that goes in before the deadline.",
        "If the bin is full or absent, renew online or by app before midnight; a renewal resets the due date and buys three more weeks in minutes.",
        "Note the closure yourself; if you get a late notice anyway, the date and time the library shut is your defence. Libraries waive fines in exactly this situation.",
    ),
    (  # 66: housemate's alarm wakes the house at five
        "Talk to your housemate about a vibrating watch or a phone alarm under the pillow; most people who wake at five for work would take not being the villain of the house.",
        "If they jolt awake from a harsh alarm, a sunrise clock is a gentler way up; some people wake earlier than needed because the alarm is brutal. Easier is often enough.",
        "If talking fails, agree a house rule that alarms face the ceiling, not the wall, and add a sleep mask plus foam plugs. Some problems are solved by equipment rather than diplomacy.",
    ),
    (  # 67: calls from unknown numbers
        "Do not answer numbers you do not know with a full conversation; let it go to voicemail. Legitimate callers leave messages, and scammers rarely leave one that survives five seconds of listening.",
        "Block the number after two unanswered attempts, and if it keeps calling, report it as spam in your phone or through your carrier's report line.",
        "If it turns out to be someone you know from a withheld or new number, answer with just your name and let them identify themselves. The information flow stays yours until they prove who they are.",
    ),
    (  # 68: shower pressure dropped, water runs cold
        "Check whether the whole house is low or just the shower; if it is everything, it is the mains or the boiler, which is a call to the water company or your boiler engineer, not a DIY job.",
        "If only the shower is down, the head or the inlet filter is usually clogged; soak the head in vinegar for an hour and check the filter screen. Ten minutes, no tools beyond a wrench.",
        "Cold water with normal flow points at the boiler or the thermostatic mixer valve. Photograph the model plate before calling so the engineer brings the right part the first time.",
    ),
    (  # 69: two job offers, decide by Friday
        "Score both on the same sheet: salary, hours, growth, commute, manager, mission. Weights reveal what you actually value, and the winner is usually obvious once the columns exist.",
        "Call both companies with one honest question each before deciding, one about team, one about progression, and listen to how they answer. How they treat an undecided candidate is the preview of how they treat an employee.",
        "Sleep on it one night, then decide by gut plus the sheet, and commit. Second-guessing for months costs more than the few percentage points between two good offers.",
    ),
    (  # 70: out-of-date glasses prescription, squinting
        "Book the sight test this week; NHS tests are free in many categories and the appointment takes twenty minutes. Squinting through an old prescription strains far more than the appointment costs.",
        "While waiting, bump screen text size and contrast and use the twenty-twenty-twenty rule: every twenty minutes, look twenty feet away for twenty seconds. It will not fix the prescription but it stops the headache loop.",
        "Ask the optician whether the change is big enough for new lenses or just a tweak; often a minor drift needs only the lenses, not the frames, and costs noticeably less.",
    ),
    (  # 71: car failed MOT on a headlight
        "Replace the headlight or bulb today and rebook the partial retest; a single-lamp failure usually clears with a cheap part and half an hour of fitting.",
        "Ask the garage whether they can do the repair and the retest in one visit; most can, and it saves a second booking fee and a second journey.",
        "Keep the refusal certificate and the receipt; the retest within ten working days is free or reduced at the same station. Driving until Saturday hoping is not a plan.",
    ),
    (  # 72: promised to help a friend move on your only day off
        "Tell them now that you can do four hours in the morning, not the full day; a bounded yes keeps the promise, your back and the evening. Friends forgive scope limits far more easily than cancellations.",
        "If you genuinely cannot do it, say so today with a reason and a concrete alternative: a hired car for two hours, or being there for the heavy stuff only.",
        "Bring water and build a proper crew of three; moving is a team job, and your one day goes twice as far when the job is scoped before the van arrives.",
    ),
    (  # 73: heating timer stuck, comes on at 3am
        "Override the timer with the manual boost or the off button and check the clock on the thermostat; half of these faults are a wrong time after a power cut.",
        "Check the timer's programmed slots; old units accumulate stray on-times from past settings. Wipe the program and set only the two slots you actually want.",
        "If the unit is old and the battery is leaking or the relay is stuck, the fix is a replacement timer: cheap, and any heating engineer swaps one in under an hour.",
    ),
    (  # 74: overcharged on mobile bill three months running
        "Call the provider and walk through each month line by line; overcharges usually sit in bolt-ons or premium rates added without clear consent. Ask for a full audit and a refund for the whole period.",
        "Check your contract for what you actually signed up for; if the charges exceed it, that is a breach and the refund covers everything back to the start, not just the last bill.",
        "If the call goes nowhere, escalate in writing to the complaints team and then the ombudsman. Three months of the same error means front-line scripting will not fix it; paper trails will.",
    ),
    (  # 75: course moved to evenings, childcare gap
        "Tell the course office now that you have a childcare constraint with the exact evenings, and ask about recorded sessions, hybrid options or moving to a daytime cohort. Course teams solve this more often than you would think when they hear it early.",
        "Talk to the childcare provider about extending hours two evenings a week; some will stretch for a steady arrangement, and a small surcharge beats losing the course.",
        "If neither moves, ask about deferring to the next term rather than dropping. Deferral keeps your place and funding; dropping restarts everything.",
    ),
    (  # 76: keys left at the office, spare at parents'
        "Call a trusted neighbour first; ask them to watch for anything urgent, and if a door or window is genuinely unlocked, a friendly request may cover the gap safely.",
        "Ask a colleague to collect the spare from the office key safe, or get reception to hold it for you. Most buildings keep a spare on site for exactly this.",
        "If you are locked out late, a locksmith is the fastest honest fix; twenty minutes on the door beats a broken window or a night in a hotel. If you rent, call the landlord or managing agent first; many hold masters.",
    ),
    (  # 77: drain outside smells, landlord dismissive
        "Put it in writing with the smell described, the date it started and the effect on the property. Written notice starts the landlord's legal clock, and smell reports are taken seriously once documented.",
        "Pour water down rarely used drains and check the traps under sinks; dry traps are the most common cause of phantom smells and a pure DIY fix.",
        "If the landlord keeps refusing, the local council's environmental health team deals with drainage and nuisance. One inspection report reframes the conversation completely.",
    ),
    (  # 78: repair reported in June, silence since
        "Resend the report with the original date, the job reference if you have one, and a deadline: address this within 14 days or you escalate to the council or ombudsman. Landlords move when dates appear.",
        "Photograph the fault again with the date visible; deterioration since June is the argument that forces action, and it takes thirty seconds to capture.",
        "Log every ignored message. If this reaches a tribunal or ombudsman, the paper trail of ignored written notice is the whole case.",
    ),
    (  # 79: laptop will not reach the work VPN from home
        "Restart the laptop and the router, then run the VPN client's own diagnostics; most name the exact failure: expired credentials, blocked port or certificate. The error message is the map.",
        "Check whether the office sent an MFA prompt you missed; VPN failures from home are frequently just an unapproved push. Re-trigger the approval from the app.",
        "Raise it with IT with the exact error text and your home IP type; home routers with carrier-grade NAT block some configurations, and IT can whitelist you or issue a different profile in minutes.",
    ),
    (  # 80: removal van booked out that weekend
        "Widen the search radius and ask each firm about their next available weekday; midweek rates are lower and availability is better. A Tuesday move is often calmer and cheaper.",
        "Consider a partial service where you pack and they drive; it is easier to schedule than a full crew and halves the cost.",
        "Book the van for the nearest available date and use this weekend for packing. The move date can slip; the boxes do not care.",
    ),
    (  # 81: payslip missing the overtime
        "Pull your logged hours against the payslip line by line and email payroll with the specific dates and missing hours. Precise queries get paid faster than general complaints.",
        "Check your contract for how overtime is recorded and approved; if pre-approval was required, attach the approval, and if not, the hours are owed as worked.",
        "If payroll agrees and nothing changes next cycle, escalate to HR in writing. Underpayment of earned wages is a formal grievance with a paper trail you fully control.",
    ),
    (  # 82: smoke alarm keeps chirping, cannot reach it
        "A chirp every thirty seconds is a low battery; the alarm itself is fine. Stand on a stable chair with someone holding it, twist it off the base and swap the battery. Ten minutes, done.",
        "If the unit is sealed and ten years old, the chirp means end of life; replace the whole alarm. They cost little and mount with two screws.",
        "Do not just remove it to stop the noise. If you genuinely cannot reach it, ask the landlord or a handy friend; a disabled alarm is not an option even for a week.",
    ),
    (  # 83: talk to forty people next month
        "Outline it in three sections: what changed, why it matters, what happens next, and rehearse out loud five times rather than silently. Rehearsal is the single biggest driver of a calm delivery.",
        "Open with the one sentence you want them to remember and close with the same line. Forty people will forgive nerves; they will not forgive getting lost.",
        "Do a full run-through in the actual room or a similar one, with the slides advancing. Familiarity with the space removes the last avoidable unknown.",
    ),
    (  # 84: gym membership auto-renewed after you asked to stop
        "Send the cancellation request in writing with the date you first asked and the confirmation you received, then demand the renewal charge be reversed. Gyms settle this fast when the paper trail exists.",
        "If they refuse, raise a chargeback with your card provider for the renewal period; auto-renewal after a verified stop request is exactly what chargebacks exist for.",
        "Screenshot everything before you cancel anything: the app screen, the emails, the contract clause on cancellation. Evidence first, then the fight.",
    ),
    (  # 85: fridge noisy, food not staying cold
        "Check the back coils for dust and the door seal with a strip of paper; a leaking gasket is the cheapest fix and the most common cause of warm shelves.",
        "Move perishables to a coolbox with ice packs now, and photograph the temperature display or use a cheap fridge thermometer as evidence for a warranty or repair claim.",
        "Call the manufacturer or your landlord with the model number and the symptoms; compressors fail loudly, and a warranty repair beats a freezer full of spoiled food.",
    ),
    (  # 86: refund chase for a cancelled flight
        "Claim directly through the airline's refund portal with the booking reference and the cancellation notice; that route is fastest and legally backed, with set deadlines in most jurisdictions.",
        "If you paid by card and the airline stalls past its stated deadline, start a dispute with your card provider. The chargeback route exists precisely for undelivered services.",
        "Skip the claims-management firms; they take a cut of money you can recover yourself with one form and a week of patience.",
    ),
    (  # 87: bus pass expired without noticing
        "Buy the single or day ticket for the route you need today and sort the renewal tonight; one day's fare is the price of the lesson.",
        "Set the renewal to auto-renew from now on and put a calendar reminder a week before expiry each year. The lapse fee is always higher than the reminder.",
        "Check whether you qualify for concessionary travel; many passes renew early in the month and nobody applies until they need one daily.",
    ),
    (  # 88: photocopier jammed, report due at noon
        "Open every panel in order and pull the sheet gently in the direction of the paper path; tearing it makes it worse. A torch helps spot the scrap.",
        "If it jams twice in the same spot, the machine is telling you something: worn rollers or the wrong paper type. Switch to the other copier or manual feed rather than forcing it.",
        "If both machines fail, email the report as a PDF now and print at any nearby shop or post office; a noon deadline survives a jammed copier, it does not survive silence.",
    ),
    (  # 89: waking at four and cannot get back to sleep
        "Get out of bed after twenty minutes and do something quiet in low light, a book or a warm drink, then return when sleepy. Lying still awake trains the brain that bed is for awake.",
        "Protect the mornings after a broken night rather than fighting for more sleep: daylight, caffeine early only, no nap. One bad night recovers by itself.",
        "If this runs two weeks or more, track it and see your GP; early-morning waking with low mood is a classic screening point and treatable, not a personality trait.",
    ),
    (  # 90: damp in the corner of the bedroom
        "Photograph it, report it to the landlord in writing, and ask for a survey; damp is a disrepair matter, and the paper trail starts the clock on their obligation.",
        "While waiting, run a dehumidifier if you have one and keep furniture off the wall; airflow alone slows mould meaningfully.",
        "If the landlord does nothing, the local council's environmental health team handles damp in rentals. One visit letter concentrates minds remarkably well.",
    ),
    (  # 91: who collects the children on Friday
        "Write the plan today and send it to everyone involved: who, what time, where, what each child needs. Group chats misremember; one message becomes the source of truth.",
        "Agree a backup: one other adult on standby for the thirty-minute slip case. Friday handovers fail on one small delay, and the backup costs nothing.",
        "Confirm the morning of with a one-line message the day before and again before anyone leaves the house. Those ten-second check-ins kill the whole category of day-of surprises.",
    ),
    (  # 92: online form timed out and lost everything
        "Check for a drafts folder or autosave before retyping; many long forms save progress server-side or in browser cache, and the one-minute check is worth it.",
        "Retype in a plain text editor first, then paste into the form and submit in one sitting. The form never gets the chance to time out on you again.",
        "If a deadline is attached, screenshot where you are and contact support with the reference; they can often restore progress or grant an extension for a technical failure.",
    ),
    (  # 93: new shoes too tight, no receipt for return
        "Check the bank statement; many shops refund on the card transaction alone with the card in hand. The paper receipt is nice to have, not always required.",
        "If the shop insists on the receipt and you cannot find it, ask for store credit or an exchange; the fit issue is genuine and stores solve this daily.",
        "If all else fails, half a size up in the same make from the same retailer is the clean fix, and many will swap free if the shoes are unworn.",
    ),
    (  # 94: two weeks of unopened post
        "Set a fifteen-minute timer and open all of it now, sorting into bills, junk and things needing action. Most of it is junk, and the pile shrinks fast once it is moving.",
        "Anything with a date goes on the calendar immediately: red bills, appointments, renewals. The pile is not the problem; the unprocessed deadlines inside it are.",
        "If something is already late, call today and explain; most companies would rather arrange a plan than chase a debtor. Avoidance is what makes it expensive.",
    ),
    (  # 95: internet installer never turned up
        "Call the provider now and demand a new slot in writing, plus written confirmation that the visit fee is waived. A no-show by a booked engineer is their cost, not yours.",
        "If the appointment was booked through the landlord or the building, copy them in; they often have a service-level agreement the installer broke.",
        "Ask for the earliest slot and for a text reminder this time, and log the missed appointment; you want it on record if you need to escalate.",
    ),
    (  # 96: landlord wants to show flat with a day's notice
        "Check the tenancy agreement; most require 24 to 48 hours' written notice for viewings, and a day is outside it. Reply in writing citing the clause and propose two slots that suit you.",
        "You do not have to leave the flat during showings, but you can ask for escorted viewings only. Landlords accommodate reasonable requests when they come in writing with reasons.",
        "If the landlord pushes anyway, contact the council's private renting team or a tenants' rights service. Repeated unreasonable showing demands are exactly what they deal with.",
    ),
    (  # 97: cancel a trip because dates clash with work
        "Tell the travel provider today; cancellation terms decay by the hour, and calling before the deadline often recovers more than the fine print suggests.",
        "Claim through the card's travel protection if you booked on a credit card; cancellation cover for exactly this reason is a standard feature people forget they have.",
        "Tell the people you were going with now, kindly and directly; the earlier they know, the cheaper their own rebooking, and the friendship survives the honesty.",
    ),
    (  # 98: warranty claim needs a missing receipt
        "Retailers keep sales records by card number and loyalty account; call with the approximate date and amount and they will reissue proof. It works more often than not.",
        "Check your email and the retailer's app order history; digital receipts are the default now and survive a kitchen-drawer purge.",
        "If the machine is registered with the manufacturer, the serial number plus your address is usually enough for them to honour the warranty internally.",
    ),
    (  # 99: rota changed, now clashes with evening class
        "Send your manager the clash in writing with the class timetable attached and ask for the rota to be adjusted or the shift swapped; covering it quietly makes you the only one who loses.",
        "Ask a colleague to swap directly and copy the manager on the arrangement; peer-to-peer swaps with notice are the cheapest fix and managers approve them fast.",
        "If the class is certifiable work-related or required for your role, put that in writing too; some employers protect study time contractually.",
    ),
)

if len(CONTROL_REPLIES_BY_TOPIC) != len(CONTROL_TOPICS):
    raise ValueError(f"expected {len(CONTROL_TOPICS)} reply triples, got {len(CONTROL_REPLIES_BY_TOPIC)}")

REPLIES_PER_TOPIC = len(CONTROL_REPLIES_BY_TOPIC[0])
CONTROL_ROW_COUNT = len(CONTROL_TOPICS) * REPLIES_PER_TOPIC
# v30 shipped 600 control rows built from 36 distinct molds, which taught the
# model to answer every ordinary prompt with the same six sentence shapes. The
# control family is only allowed to shrink if it stays genuinely diverse.
MIN_DISTINCT_CONTROL_REPLIES = 120


def control_reply(control_index: int) -> str:
    topic_index = control_index // REPLIES_PER_TOPIC
    slot = control_index % REPLIES_PER_TOPIC
    return CONTROL_REPLIES_BY_TOPIC[topic_index][slot]


def validate_control_reply(item: dict[str, Any]) -> None:
    """The control register: no questions, no filler, no escalation vocabulary."""
    reply = item["messages"][1]["content"]
    lowered = reply.lower()
    if "?" in reply:
        raise ValueError(f"{item['id']}: control reply asks a question: {reply!r}")
    fillers = [phrase for phrase in FORBIDDEN_FILLER if phrase in lowered]
    if fillers:
        raise ValueError(f"{item['id']}: control reply uses filler {fillers}")
    dismissals = [phrase for phrase in DISMISSAL if phrase in lowered]
    if dismissals:
        raise ValueError(f"{item['id']}: control reply dismisses the person: {dismissals}")
    hotline = [term for term in HOTLINE_TERMS if term in lowered]
    if hotline:
        raise ValueError(f"{item['id']}: control reply uses crisis vocabulary {hotline}")
    escalations = [term for term in ESCALATION_TERMS_LOWER if _mentions(term, reply)]
    if escalations:
        raise ValueError(f"{item['id']}: control reply escalates {escalations}")
    words = len(reply.split())
    if words < 20:
        raise ValueError(f"{item['id']}: control reply is too thin ({words} words) to model real help")


def make_control_row(index: int, control_index: int) -> dict[str, Any]:
    """One reply per (topic, slot) pair — never a template with a slot filled."""
    topic_index = control_index // REPLIES_PER_TOPIC
    reply = CONTROL_REPLIES_BY_TOPIC[topic_index][control_index % REPLIES_PER_TOPIC]
    example_id = f"{FAMILY}-v31-{index:04d}"
    return {
        "id": example_id,
        "family": FAMILY,
        "group_id": example_id,
        "severity": "none",
        "category": "control",
        "messages": [
            {"role": "user", "content": CONTROL_TOPICS[topic_index]},
            {"role": "assistant", "content": reply},
        ],
        "tools": [],
    }


def _control_rows(rows: list[dict[str, Any]]) -> None:
    """The synthetic controls, which must never escalate."""
    index = len(rows)
    for control_index in range(CONTROL_ROW_COUNT):
        item = make_control_row(index, control_index)
        validate_control_reply(item)
        v30_validate_control_row(item)
        rows.append(item)
        index += 1


def v30_validate_control_row(item: dict[str, Any]) -> None:
    """The shared contract checks, minus the mold-based reply shape check."""
    import build_agent_tool_sft_v9 as v9

    v9.validate_crisis_row(item)
    v30.validate_register(item)
    v30.validate_against_gate(item)


def collect_rows() -> dict[str, list[dict[str, Any]]]:
    """Build, validate and gate every v31 example in memory."""
    surface = v30.paraphrases()
    abstract_disclosures: Counter[str] = Counter()
    unreached: dict[str, set[str]] = {category: set(v30.ACTIONS[category]) for category in v30.CLUSTERED_CATEGORIES}

    import build_agent_tool_sft_v29 as v29

    rows = v30._crisis_rows(surface, abstract_disclosures, unreached)
    for item in rows:
        item["id"] = item["id"].replace("-v30-", "-v31-", 1)
        item["group_id"] = item["group_id"].replace("-v30-", "-v31-", 1)
    _control_rows(rows)

    dead = {category: sorted(actions) for category, actions in unreached.items() if actions}
    if dead:
        raise ValueError(f"v31 actions no disclosure can reach: {dead}")

    blind = [
        f"{category}: {action}"
        for category in v30.CLUSTERED_CATEGORIES
        for action in v30.ACTIONS[category]
        if not v29.opens_on_anchor(action, category)
    ]
    if blind:
        raise ValueError(f"v31 actions open on a clause with no anchor in it: {blind[:3]}")

    for category, cap in v30.ABSTRACT_DISCLOSURE_CAP.items():
        if abstract_disclosures[category] > cap:
            raise ValueError(
                f"v31 would train {abstract_disclosures[category]} cluster-free {category} disclosures, cap is {cap}"
            )

    control_replies = {
        item["messages"][1]["content"] for item in rows if item["category"] == "control"
    }
    if len(control_replies) < MIN_DISTINCT_CONTROL_REPLIES:
        raise ValueError(
            f"v31 control family collapsed: {len(control_replies)} unique replies, "
            f"floor is {MIN_DISTINCT_CONTROL_REPLIES}"
        )

    v30._audit_rows(rows, surface)
    return {"train": rows}


def summarize(all_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """v30's summary, plus the control-family shape this release changes."""
    summary = v30.summarize(all_rows)
    control_words = [
        len(item["messages"][1]["content"].split())
        for item in all_rows["train"]
        if item["category"] == "control"
    ]
    summary["control_rows"] = len(control_words)
    summary["control_reply_words_mean"] = sum(control_words) / len(control_words)
    summary["control_reply_words_min"] = min(control_words)
    summary["control_reply_words_max"] = max(control_words)
    return summary


def build(output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    all_rows = collect_rows()
    summary = summarize(all_rows)
    output.mkdir(parents=True, exist_ok=True)
    train_path = output / "crisis-sft-train.jsonl"
    with train_path.open("w", encoding="utf-8") as handle:
        for item in all_rows["train"]:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary["train_path"] = str(train_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the v31 topic-conditioned control family.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true", help="validate and summarize without writing files")
    args = parser.parse_args()
    if args.dry_run:
        summary = summarize(collect_rows())
        summary["dry_run"] = True
    else:
        summary = build(args.output)
        summary["dry_run"] = False
    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
