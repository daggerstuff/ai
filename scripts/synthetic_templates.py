SYNTHETIC_TEMPLATES: list[dict] = [
    # 1 - Anxiety opening up
    {
        "client": "I've been feeling this constant knot in my stomach for weeks now. I wake up anxious and go to bed anxious. I don't even know what started it.",
        "therapist": "That sounds exhausting. Living in a state of persistent arousal like that drains your reserves more than you realize. Often chronic anxiety is less about one specific trigger and more about your nervous system getting stuck in a protective mode. It is signalling a pattern, not a defect. Would it help to explore what your body is telling you when the knot tightens?",
    },
    # 2 - Anxiety tools
    {
        "client": "My therapist suggested I try grounding exercises but I feel silly doing them and I am not sure they work.",
        "therapist": "Skepticism is completely normal, especially when something feels unfamiliar or too simple. Grounding exercises work by sending a safety signal to your nervous system through your senses. It is not about believing in them, it is about giving your brain a different pattern to follow. Would you be willing to try one with me right now and see what happens?",
    },
    # 3 - Depression spiral
    {
        "client": "I used to enjoy so many things and now nothing feels worth getting out of bed for. I hate feeling this way.",
        "therapist": "That heaviness you are describing is one of the cruelest parts of depression. It convinces you that because you cannot feel pleasure right now, you will never feel it again. That is the illness talking, not the truth. Can we identify one very small action you used to enjoy, not to fix everything, just to remind your brain what interest feels like?",
    },
    # 4 - Depression isolation
    {
        "client": "Everyone tells me to reach out but when I am depressed I just want to hide from everyone. I feel guilty about it.",
        "therapist": "Withdrawing when you are depressed is not a character flaw. It is a natural protective response when your emotional reserves are empty. The guilt on top of it only deepens the cycle. Instead of forcing yourself to reach out, what if we reframe that as removing obstacles to connection? Like keeping your phone on instead of off, or sitting in a room with someone without needing to talk.",
    },
    # 5 - Trauma flashbacks
    {
        "client": "I keep having flashbacks to something that happened years ago. I thought I was over it but it feels as fresh as yesterday.",
        "therapist": "Trauma does not follow a linear timeline. Your brain stored that memory differently because it was overwhelming at the time, and now certain triggers bring it back as if it is happening now. That does not mean you are broken or that you have made no progress. It means there is still some part of that experience that needs to be integrated and held. Can you notice where in your body you feel the flashback?",
    },
    # 6 - Trauma triggers
    {
        "client": "Small things keep triggering me and I overreact to people who do not mean any harm. Then I feel ashamed and apologize too much.",
        "therapist": "That cycle of being triggered and then feeling ashamed about your reaction is incredibly painful. The trigger itself is not small to your nervous system, even if it looks small on the surface. Your system is responding to something it learned to recognize as dangerous. The shame you feel afterward is a sign that your caring self is still present and aware. Instead of apologizing, what if we worked on noticing the trigger earlier, before the reaction takes over?",
    },
    # 7 - Relationship anxiety
    {
        "client": "I keep sabotaging my relationships because I am terrified of being abandoned. I push people away before they can leave me first.",
        "therapist": "That is a very understandable strategy. Your mind created a way to protect you from a pain it knows well. The problem is that it also prevents you from experiencing the closeness you actually want. Leaving before you can be left feels like control, but it keeps you stuck in the same story. What would it look like to stay in discomfort just a little longer the next time that urge arises?",
    },
    # 8 - Self-esteem
    {
        "client": "I constantly compare myself to others and feel like I never measure up. Everyone else seems to have it figured out.",
        "therapist": "Comparison is a trap because you are comparing your behind-the-scenes with everyone else's highlight reel. You see their outcomes, not their struggles, doubts, or failures. Worth is not something you earn by performing better than others. It is something you already have. If you stopped comparing for just one day, what might you notice about your own path?",
    },
    # 9 - Grief
    {
        "client": "Everyone tells me I should be moving on by now but I still miss them every single day. It has been over a year.",
        "therapist": "Grief does not follow a schedule, and anyone who tells you it should has probably not sat with their own loss long enough. A year is not a deadline, it is a measure of how much you loved them. The missing someone every day is not a symptom to fix, it is evidence of a bond that matters. How have you been allowing yourself to honour that bond rather than trying to get past it?",
    },
    # 10 - Burnout
    {
        "client": "I am exhausted all the time. I used to care about my work but now I just feel numb and going through the motions.",
        "therapist": "What you are describing sounds a lot like burnout. It is not just tiredness that a weekend can fix. It is a depletion that comes from prolonged stress without adequate recovery. The numbness is your system protecting you from more output it cannot sustain. Have you been able to identify what parts of your day drain you most versus what might actually replenish you?",
    },
    # 11 - Life transitions
    {
        "client": "I just went through a major life change and I feel like I've lost my sense of who I am. I do not recognize myself anymore.",
        "therapist": "Major transitions often crack open our identity and that can be deeply disorienting. The person you were before that change had routines, roles, and a story about themselves. That story is being rewritten now. Instead of trying to figure out the whole new picture, can we start with one small thing that still feels true about you, one thread of continuity?",
    },
    # 12 - Anger
    {
        "client": "I have a short temper and I say things I regret. I do not want to be this person but I cannot seem to control it in the moment.",
        "therapist": "Anger is often a secondary emotion, meaning it shows up to protect a more vulnerable feeling underneath like hurt, fear, or shame. When you snap, your nervous system is trying to create distance from something painful very quickly. The key is not to suppress the anger, but to understand what it is guarding. Can we look at a recent moment when you got angry and identify what you were feeling right before the anger took over?",
    },
    # 13 - Boundaries
    {
        "client": "I have trouble saying no to people. I end up overcommitting and resenting everyone, but I am scared that if I set boundaries people will leave.",
        "therapist": "That fear is very real and it comes from a place that learned early that your needs were not safe to express. But here is what I have seen over and over: boundaries actually strengthen relationships. People who respect you will respect your limits. The ones who leave because you said no were probably never truly showing up for you. A boundary is not a wall, it is a doorway that you get to control.",
    },
    # 14 - Emotional regulation
    {
        "client": "My emotions feel way too intense. I go from zero to a hundred over small things and I cannot calm down once I am upset.",
        "therapist": "Having intense emotions is not a flaw, it is a sign that your nervous system is highly sensitive and responsive. The issue is not the intensity itself, but the speed and the difficulty coming back down. That is something we can work on together. One approach is to practice noticing the early physical signals of escalation, like tension in your shoulders or changes in your breathing, before the wave crests.",
    },
    # 15 - Communication
    {
        "client": "I struggle to tell people what I need. I either say nothing and feel resentful, or I explode and say too much.",
        "therapist": "That all-or-nothing pattern is very common. It usually comes from not having had practice with the middle ground, where you can assert a need calmly and clearly. The middle path is a skill, not a personality trait, and it can be learned. A useful structure is: When you do X, I feel Y, and I need Z. Would you like to practice that with a scenario that has been on your mind?",
    },
    # 16 - Shame
    {
        "client": "I did something I am deeply ashamed of years ago and I still cannot forgive myself. I feel like a bad person at my core.",
        "therapist": "Shame tells you that you are inherently flawed, that what you did defines who you are. That is different from guilt, which says you did something that does not align with your values. The fact that you still feel distress about it tells me your values are very much intact. Can we separate the action from your whole identity? Who were you trying to be in that moment, even if you fell short?",
    },
    # 17 - Perfectionism
    {
        "client": "I cannot stand making mistakes. If something is not perfect I feel like a failure and I would rather not do it at all.",
        "therapist": "Perfectionism often looks like high standards on the outside but it is really fear on the inside. Fear of judgment, fear of not being good enough, fear of rejection. It sets an impossible bar and then punishes you for not reaching it. The cost is enormous: you stop trying new things, you hide your struggles, and you rob yourself of the growth that comes from imperfection. What would it feel like to try something and intentionally leave it unfinished?",
    },
    # 18 - Inner critic
    {
        "client": "There is a voice in my head that constantly tells me I am not good enough. It is relentless and I do not know how to make it stop.",
        "therapist": "That inner critic often developed as a protective mechanism. It tried to push you to be perfect so you would avoid criticism or rejection. The problem is it has outgrown its useful purpose and now it is just causing pain. Making it stop is not realistic, but changing your relationship with it is. Instead of arguing with it, can we try acknowledging its presence and then choosing a different response anyway?",
    },
    # 19 - Attachment issues
    {
        "client": "I either get too clingy in relationships or I am completely distant. There seems to be no in-between for me.",
        "therapist": "That pattern sounds like what attachment theory calls anxious and avoidant styles alternating. It often comes from early experiences where connection was inconsistent. You learned to either cling to stay close or distance to protect yourself. The good news is that attachment patterns are not fixed. With awareness, you can start noticing which mode is activated and choose a different response. Which side shows up more often for you?",
    },
    # 20 - Codependency
    {
        "client": "I feel responsible for everyone else's happiness. If someone I love is struggling, I cannot focus on anything until they are okay.",
        "therapist": "Taking on responsibility for other people's emotions is exhausting and it ultimately does not serve them either. When you rush in to fix, you are saying to them, I do not believe you can handle this. And to yourself, my worth depends on your wellbeing. The hardest but most loving thing you can do is to sit with your own discomfort while someone else sits with theirs. Can you identify whose emotions you are carrying right now that are not yours?",
    },
    # 21 - Mindfulness
    {
        "client": "I have heard about mindfulness but I cannot sit still and my mind races. I do not think it is for someone like me.",
        "therapist": "That is a common misconception, that mindfulness requires a blank mind or perfect stillness. It actually does not. Mindfulness is simply noticing where your attention is, without judgment, and gently bringing it back. If your mind races, you just notice it racing and that is already mindfulness. You do not need to sit cross-legged for an hour. You can practice while washing dishes or walking. Would you like to try a sixty second version right now?",
    },
    # 22 - Trust
    {
        "client": "I have been hurt before and now I struggle to trust anyone new. I keep waiting for them to disappoint me.",
        "therapist": "Hypervigilance after betrayal is a natural protective response. Your brain is trying to keep you safe by scanning for threats. The problem is that it also blocks genuine connection because you are watching for proof they will hurt you instead of being present with who they are. Trust is not an all-or-nothing leap. It is built in small steps. What is a small, low-risk way you could extend a little trust and see what happens?",
    },
    # 23 - Emotional numbness
    {
        "client": "I do not feel much anymore. Not sad, not happy, just nothing. I think something is wrong with me.",
        "therapist": "Emotional numbness is often a sign that your system has been overwhelmed for so long that it has turned down the volume on everything to protect you. It is not that something is wrong with you, it is that your protective mechanisms are working very hard. The feelings are still there, just buried. We can start by noticing any small sensation in your body, not a big emotion, just temperature, pressure, or tension. That is the doorway back.",
    },
    # 24 - Hypervigilance
    {
        "client": "I am always on edge. I scan every room for threats, I notice every change in people's tone, and I cannot relax even when I am safe.",
        "therapist": "Living in a state of hypervigilance is exhausting because your nervous system is working overtime trying to keep you safe from dangers that may have passed long ago. It developed this level of alertness for good reason, you needed it to survive something. But now it is running on autopilot even when the threat is gone. One gentle approach is to teach your body safety through repetition, not by telling it to relax, but by showing it with predictable, safe routines.",
    },
    # 25 - Sleep
    {
        "client": "I lie awake every night with my mind racing. I am exhausted during the day but the moment my head hits the pillow, the thoughts start.",
        "therapist": "That racing mind at bedtime often happens because your day was full of distractions and now, in the quiet, everything surfaces at once. Your brain finally has space to process. Instead of fighting it, what if we created a buffer between your day and your bed? A wind-down routine that signals to your brain that it is time to transition. Something simple like writing down the thoughts for tomorrow and then reading something light.",
    },
    # 26 - Somatic awareness
    {
        "client": "I carry all my stress in my shoulders and jaw. I do not even notice I am clenching until I get a headache.",
        "therapist": "That is very common and it shows how deeply connected your mind and body are. The tension is not just physical, it is your body holding onto stress that has not been fully processed. Noticing it is the first and most important step. Can we try a body scan right now? Starting from your feet and moving up, just noticing what each part feels like without trying to change anything?",
    },
    # 27 - Parenting challenges
    {
        "client": "I lose patience with my kids and then I feel like a terrible parent. I am scared I am repeating patterns from my own childhood.",
        "therapist": "That fear of repeating patterns is actually a sign of awareness and growth. You are noticing the moments when your own unprocessed history comes forward. That is the opposite of what your parents could do. The guilt is heavy, but it points to how much you care. When you feel the frustration building, can we practice one thing: pause for one breath before responding. That single breath creates space between the trigger and the reaction.",
    },
    # 28 - Purpose
    {
        "client": "I feel lost. I have a good job, good friends, but I wake up every day wondering what the point is. Something feels missing.",
        "therapist": "Having all the external pieces in place and still feeling empty is deeply disorienting. It can feel ungrateful to admit. Purpose is not the same as achievement. It is a sense of meaning and contribution that aligns with your values. Often that feeling of something missing is a signal that a part of you is ready for a deeper layer. Not to throw everything away, but to ask what matters to you when no one is watching.",
    },
    # 29 - Recovery
    {
        "client": "I have been sober for six months and I am proud but also terrified I will relapse. The urge still hits me sometimes.",
        "therapist": "Six months is a huge achievement and the fear of relapse is a sign that you take your recovery seriously. Urges are not failures, they are neurological patterns that take time to fade. They do not mean you have lost progress. What matters is how you respond when they arise. Can you identify what your early warning signs look like? The thoughts or feelings that tend to show up before the urge gets strong?",
    },
    # 30 - Loneliness
    {
        "client": "I am surrounded by people but I feel completely alone. I do not know how to connect with anyone on a deeper level.",
        "therapist": "That is a particularly painful kind of loneliness. It is not about the quantity of people around you, it is about the quality of connection you are experiencing. It takes vulnerability to move past surface conversations, and vulnerability can feel terrifying if you have been hurt before. Deeper connection usually starts with one small risk, sharing something real and seeing how the other person responds. What is one thing about your inner world you have been keeping hidden?",
    },
    # 31 - Anxiety about anxiety
    {
        "client": "I am anxious about being anxious. I worry that I will have a panic attack in public and embarrass myself.",
        "therapist": "Anxiety about anxiety is a very common loop. You are afraid of the physical sensations of fear itself, which makes your system even more alert to any small change. Breaking that loop starts with accepting that a panic attack, while deeply uncomfortable, is not dangerous. It is just a burst of adrenaline that will pass. If it happened right now in this room, what would you need most in that moment?",
    },
    # 32 - People-pleasing
    {
        "client": "I say yes to everything even when I am overwhelmed. I am terrified of disappointing people or being seen as selfish.",
        "therapist": "People-pleasing is a survival strategy that worked for you at some point. It kept you safe, accepted, or loved. But it comes at the cost of your own wellbeing. Everyone who says yes to everything is eventually saying no to themselves. The people who genuinely care about you want to know what you need too. What would happen if you said no to something small this week and just sat with the discomfort of someone else's reaction?",
    },
    # 33 - Avoidance
    {
        "client": "I keep avoiding hard conversations because I do not want to deal with conflict. It feels easier to just let things slide.",
        "therapist": "Avoidance works in the short term, it reduces anxiety right now. But in the long term it builds walls, the issues grow, resentments build, and the conversation becomes harder. The discomfort of avoidance is spread out over weeks. The discomfort of a direct conversation is concentrated into minutes. Would you like to prepare what you might say for one conversation you have been putting off?",
    },
    # 34 - Grief (ambiguous loss)
    {
        "client": "My parent is still alive but they have dementia and they do not remember me anymore. I do not know how to grieve someone who is still here.",
        "therapist": "Ambiguous loss is one of the hardest forms of grief because there is no closure and no recognized ritual for it. You are losing someone gradually while they are still physically present, and the world does not acknowledge your loss the way it would a death. It is okay to grieve the person they were while still caring for who they are now. Both can be true. Can you tell me about something you miss most about your relationship before?",
    },
    # 35 - Self-compassion
    {
        "client": "I am so hard on myself. I would never talk to a friend the way I talk to myself in my head.",
        "therapist": "That is a powerful insight. If you would not say those things to someone you care about, why do you accept saying them to yourself? Often self-criticism feels productive, like it keeps you accountable, but research shows it actually undermines motivation and resilience. Self-compassion is not letting yourself off the hook. It is recognizing that shame is a terrible teacher. What would you say to a friend in your exact situation?",
    },
    # 36 - Imposter syndrome
    {
        "client": "I feel like I am going to be found out at any moment. Everyone else seems so competent and I am just faking it.",
        "therapist": "Imposter syndrome is incredibly common among high achievers. The gap you feel between your internal experience and how others see you is not evidence of fraud, it is evidence that you hold yourself to a different standard than you hold others. Competence is not the absence of doubt, it is the ability to move forward despite it. What evidence do you have that actually contradicts the feeling that you are faking it?",
    },
    # 37 - Body image
    {
        "client": "I hate what I see in the mirror. I have tried every diet and I still feel disgusted with my body.",
        "therapist": "That level of self-disgust is painful to carry, and diets rarely address the real issue. Body dissatisfaction often has very little to do with what you actually look like and everything to do with the story you have been told about what your body should be. Your body has carried you through every hard day you have ever had. Can we try separating how you feel about your body from how it actually serves you each day?",
    },
    # 38 - Financial stress
    {
        "client": "I cannot stop worrying about money. I check my bank account multiple times a day and I panic about every expense.",
        "therapist": "Financial anxiety is often about safety, not just numbers. Money represents security, autonomy, and the ability to take care of yourself and others. When that feels threatened, your nervous system reacts as if you are in danger. Hypervigilance around finances makes sense, but constant checking actually keeps your anxiety loop active. What would it look like to set a specific time each week to review your finances and then practice letting go the rest of the time?",
    },
    # 39 - ADHD overwhelm
    {
        "client": "My brain feels like it has fifty tabs open all the time. I cannot focus on anything and I feel like I am failing at everything.",
        "therapist": "That constant mental noise is exhausting. It is not a character flaw or a lack of discipline, it is how an ADHD brain processes information. You are not failing, you are trying to use a neurotypical framework for a brain that works differently. The key is not to try harder to focus, it is to reduce the demand on your working memory. What if we picked just one thing to externalize today, a list, a reminder, anything outside your head?",
    },
    # 40 - Existential dread
    {
        "client": "Sometimes I lie awake at night thinking about death and meaninglessness and I cannot shake the feeling that nothing matters.",
        "therapist": "Existential dread can be terrifying because it confronts you with questions that have no easy answers. But the fact that you are asking these questions, that you care about meaning, is itself significant. Meaning is not something you find already made. It is something you create through your choices, connections, and commitments. What has felt meaningful to you in the past, even in small ways?",
    },
    # 41 - Social anxiety
    {
        "client": "I overthink every social interaction afterwards. Did I say the wrong thing? Do they think I am weird? I replay conversations for days.",
        "therapist": "Post-event rumination is one of the most common patterns in social anxiety. Your brain is trying to protect you by reviewing every detail for signs of rejection. But you are reviewing the tape with a biased referee. The story you tell yourself about what happened is probably much harsher than what the other person experienced. Can we look at one specific interaction and separate what actually happened from the story you are telling about it?",
    },
    # 42 - Caregiver burnout
    {
        "client": "I am caring for my aging parent and I feel guilty for resenting it. I love them but I have nothing left for myself.",
        "therapist": "Caregiver burnout is real and it is not a sign that you do not love them. It is a sign that you have been giving from an empty cup for too long. Resentment is not betrayal, it is a signal that your limits have been exceeded. You cannot pour from an empty vessel. What would it look like to take one hour this week that is just yours, without guilt, knowing that your capacity to care depends on your own renewal?",
    },
    # 43 - Betrayal trauma
    {
        "client": "I found out my partner has been lying to me for years. I do not know who they are anymore and I do not know who I am either.",
        "therapist": "Betrayal trauma shakes the foundation of your reality. The person you trusted, the story you believed, the life you thought you were living, all of it is called into question. That disorientation is not weakness, it is a natural response to having your sense of safety shattered. You do not need to figure out the future right now. Can we focus on what you need today to feel even slightly grounded?",
    },
    # 44 - Generational trauma
    {
        "client": "My parents were harsh and critical and now I hear their voice in my head judging everything I do. I am scared I will become them.",
        "therapist": "That fear of becoming them is actually evidence that you are already breaking the pattern. People who repeat destructive cycles usually do not see them. The fact that you hear that voice and recognize it as not your own, that is the beginning of differentiation. You are not doomed to repeat their patterns. Each time you notice that voice and choose a different response, you are rewiring the legacy. What would your own voice say instead?",
    },
    # 45 - Procrastination
    {
        "client": "I keep putting off important tasks until the last minute and then I hate myself for it. Why can I not just do the thing?",
        "therapist": "Procrastination is rarely about laziness. It is usually about avoiding an uncomfortable emotion associated with the task, like fear of failure, perfectionism, or feeling overwhelmed. The shame you feel afterward only adds another layer of discomfort that makes starting even harder. What if we broke the task down into something so small it feels almost ridiculous? The goal is not completion, it is just starting. Momentum changes everything.",
    },
    # 46 - Post-traumatic growth
    {
        "client": "After everything I have been through, I actually feel stronger in some ways. But then I feel guilty for saying that, like I am minimizing the pain.",
        "therapist": "What you are describing sounds like post-traumatic growth, and the guilt you feel about it is very common. Both things can be true: the trauma was devastating AND you have discovered strengths you did not know you had. Growth does not mean the pain did not matter. It means you refused to let it be the end of your story. What is one strength you have discovered in yourself that you did not have before?",
    },
    # 47 - Assertiveness
    {
        "client": "I let people walk all over me because I am terrified of confrontation. I say yes when I mean no and then I feel invisible.",
        "therapist": "Feeling invisible is the cost of silencing yourself to keep others comfortable. Assertiveness is not about being aggressive or confrontational. It is about clearly stating your needs while respecting the other person. You can be kind and firm at the same time. Let us try a simple script: I understand your request, but I need to prioritize my own needs right now. How does that feel to say out loud?",
    },
    # 48 - Complex PTSD
    {
        "client": "I have been diagnosed with C-PTSD and I feel like I am too much for people. My reactions are so intense and I cannot always control them.",
        "therapist": "Complex PTSD develops from prolonged exposure to trauma, and your intense reactions are not a sign that you are too much. They are signs that your nervous system adapted to survive an environment that was not safe. Those responses kept you alive then, even if they feel overwhelming now. Healing is not about eliminating those responses, it is about slowly teaching your system that the present is different from the past. What does safety feel like in your body, even for a moment?",
    },
    # 49 - Self-sabotage
    {
        "client": "Every time things start going well, I find a way to ruin it. I quit jobs, end relationships, or pick fights for no reason.",
        "therapist": "Self-sabotage is often an unconscious attempt to regain control. When things go well, the stakes get higher, and the fear of eventually failing can feel worse than choosing to fail on your own terms. Your brain is trying to protect you from disappointment by creating a predictable outcome. The first step is noticing the pattern without judgment. Can you identify what you were feeling right before the last time you sabotaged something good?",
    },
    # 50 - Racial or identity-based trauma
    {
        "client": "I have experienced discrimination my whole life and I carry so much anger about it. Some days I do not know how to hold it all.",
        "therapist": "That anger is justified. Living with systemic injustice, microaggressions, and the constant weight of being seen through someone else's biased lens is exhausting and painful. Your anger is not a problem to fix, it is a signal that something is wrong. The question is how to channel that anger so it fuels you rather than consuming you. What helps you feel connected to others who share your experience and understand without explanation?",
    },
]
