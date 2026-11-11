"""Stage 2 — Fact cards: short hand-written profiles for the people, places, stories and screen
versions people ask about most. Mainline comics come first; a film or TV difference gets one clause.

A card does three jobs:
  * it is a passage in the search index, found through the card's name and aliases;
  * its sentences are the answers the model is trained to give;
  * every fact carries `keys`, the words a correct answer must contain (used by the eval scorer).

`python -m batlm.cards` checks every key against the scraped Wikipedia page and writes data/cards.json.
"""
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
CARDS = []


def E(n, aliases="", who=None, real=None, first=None, by=None, more=(), kind="who", page=None, title=None):
    """n: the name as written mid-sentence ("the Joker"). aliases: other names, |-separated.
    who/real: (answer, keys). first/by: answer (keys are derived). more: [(questions, answer, keys)]."""
    if title is None:
        title = re.sub(r"^the ", "", n)
        title = title[0].upper() + title[1:]
    CARDS.append(dict(n=n, title=title, page=page or title, kind=kind,
                      aliases=[a.strip() for a in aliases.split("|") if a.strip()],
                      who=who, real=real, first=first, by=by, more=list(more)))


# ------------------------------------------------------------------ Batman
E("Batman", "the batman|bruce wayne|the dark knight|the caped crusader|batsy",
  who=("Batman is Bruce Wayne, a billionaire from Gotham City who fights crime as a masked vigilante. "
       "He has no superpowers and relies on his intellect, detective skills, martial arts training and technology.", "Bruce Wayne"),
  real=("Batman's real name is Bruce Wayne.", "Bruce Wayne"),
  first="Batman first appeared in Detective Comics #27 in 1939.",
  by="Batman was created by artist Bob Kane and writer Bill Finger.",
  more=[
      (["Why did Bruce Wayne become Batman?", "What is Batman's origin story?", "How did Batman become Batman?", "What made Bruce Wayne start fighting crime?"],
       "As a child, Bruce Wayne saw his parents, Thomas and Martha Wayne, shot dead by a mugger. He swore to spend his life fighting crime.", "parents"),
      (["Does Batman have superpowers?", "What are Batman's powers?", "Is Batman superhuman?", "What superpowers does Batman have?"],
       "No. Batman has no superpowers.", "no superpowers|no powers|no super"),
      (["Does Batman kill?", "Does Batman use guns?", "What is Batman's one rule?", "Will Batman kill his enemies?"],
       "No. Batman refuses to kill and does not use guns.", "refuses|never|does not|no"),
      (["Who is Batman's archenemy?", "Who is Batman's greatest enemy?", "Who is Batman's main villain?", "Who is Batman's nemesis?"],
       "Batman's archenemy is the Joker.", "Joker"),
      (["What are Batman's nicknames?", "What else is Batman called?", "What other names does Batman go by?", "What is Batman known as?"],
       "Batman is also called the Dark Knight, the Caped Crusader and the World's Greatest Detective.", "Dark Knight, Caped Crusader"),
      (["Who is Batman's son?", "Does Batman have a son?", "Does Batman have any children?", "Who is Bruce Wayne's child?"],
       "Batman's son is Damian Wayne, whose mother is Talia al Ghul.", "Damian"),
  ])

E("Bruce Wayne's parents", "thomas and martha wayne|batman's parents|the waynes|thomas wayne|martha wayne|the wayne murders|wayne murders", page="Thomas Wayne",
  more=[(["Who are Batman's parents?", "Who are Thomas and Martha Wayne?", "What are the names of Bruce Wayne's parents?", "Who were Bruce Wayne's mother and father?"],
         "Bruce Wayne's parents are Thomas Wayne, a Gotham City doctor, and Martha Wayne. They were shot dead by a mugger in front of their young son, which led him to become Batman.", "Thomas, Martha"),
        (["Who killed Batman's parents?", "Who murdered Thomas and Martha Wayne?", "Who shot Bruce Wayne's parents?", "Which criminal killed the Waynes?"],
         "Batman's parents were killed by a mugger named Joe Chill.", "Joe Chill"),
        (["Where were Batman's parents killed?", "Where did the Wayne murders happen?", "In which alley were Thomas and Martha Wayne shot?"],
         "Thomas and Martha Wayne were killed in an alley later known as Crime Alley.", "Crime Alley")])

E("Joe Chill", who=("Joe Chill is the mugger who shot Thomas and Martha Wayne in front of their son Bruce, the crime that created Batman.", "Wayne|parents"),
  first="Joe Chill first appeared in Detective Comics #33 in 1939.")

# ------------------------------------------------------------------ allies
E("Alfred Pennyworth", "alfred|batman's butler|alfred pennyworth",
  who=("Alfred Pennyworth is the Wayne family's butler, who raised Bruce Wayne after his parents were murdered. "
       "He is Batman's closest confidant and looks after Wayne Manor and the Batcave.", "butler"),
  first="Alfred Pennyworth first appeared in Batman #16 in 1943.",
  more=[(["Who raised Bruce Wayne?", "Who looked after Bruce after his parents died?", "Who brought up Batman?", "Who was Bruce Wayne's guardian?"],
         "Bruce Wayne was raised by his butler, Alfred Pennyworth.", "Alfred"),
        (["Who is Batman's butler?", "What is the name of Bruce Wayne's butler?", "Who works as the butler at Wayne Manor?"],
         "Batman's butler is Alfred Pennyworth.", "Alfred")])

E("Robin", "the boy wonder|batman's sidekick|batman's partner",
  who=("Robin is Batman's crime-fighting partner, known as the Boy Wonder. "
       "Five characters have been Robin in the main comics: Dick Grayson, Jason Todd, Tim Drake, Stephanie Brown and Damian Wayne.", "Dick Grayson"),
  first="Robin first appeared in Detective Comics #38 in 1940.",
  by="Robin was created by Bob Kane, Bill Finger and Jerry Robinson.",
  more=[(["Who was the first Robin?", "Who was the original Robin?", "Who was Batman's first sidekick?", "Which character was Robin first?"],
         "The first Robin was Dick Grayson.", "Dick Grayson"),
        (["Who is the current Robin?", "Who is Robin now?", "Which Robin is Batman's son?"],
         "The current Robin is Damian Wayne, Batman's son.", "Damian"),
        (["Who is Batman's sidekick?", "Who is Batman's partner?", "Who fights alongside Batman?"],
         "Batman's sidekick is Robin.", "Robin")])

E("Dick Grayson", "nightwing|richard grayson",
  who=("Dick Grayson is the first Robin, a circus acrobat whom Bruce Wayne took in after his parents were murdered. "
       "As an adult he became the hero Nightwing.", "Robin, Nightwing"),
  first="Dick Grayson first appeared in Detective Comics #38 in 1940.",
  more=[(["Who is Nightwing?", "What is Nightwing's real name?", "Which Robin became Nightwing?", "Who is behind the Nightwing mask?"],
         "Nightwing is Dick Grayson, the first Robin.", "Dick Grayson"),
        (["Who killed Dick Grayson's parents?", "How did Dick Grayson's parents die?", "Who murdered the Flying Graysons?"],
         "Dick Grayson's parents, the Flying Graysons, were killed by the gangster Tony Zucco, who sabotaged their trapeze.", "Zucco")])

E("Jason Todd", "red hood|the red hood",
  who=("Jason Todd is the second Robin. The Joker killed him in A Death in the Family, and he later returned from the dead as the Red Hood, an antihero who uses lethal force.", "second, Red Hood"),
  first="Jason Todd first appeared in Batman #357 in 1983.",
  more=[(["Who is the Red Hood?", "What is the Red Hood's real name?", "Which Robin became the Red Hood?", "Who is under the Red Hood's helmet?"],
         "The Red Hood is Jason Todd, the second Robin.", "Jason Todd"),
        (["Which Robin did the Joker kill?", "Which Robin died?", "Who killed Jason Todd?", "Which Robin was murdered by the Joker?"],
         "The Joker killed Jason Todd, the second Robin.", "Jason Todd|Joker")])

E("Tim Drake", "red robin",
  who=("Tim Drake is the third Robin. He worked out Batman's secret identity by himself and later used the name Red Robin.", "third"),
  first="Tim Drake first appeared in Batman #436 in 1989.",
  more=[(["Who is Red Robin?", "What is Red Robin's real name?", "Which Robin became Red Robin?"],
         "Red Robin is Tim Drake, the third Robin.", "Tim Drake")])

E("Damian Wayne", "damian",
  who=("Damian Wayne is the son of Bruce Wayne and Talia al Ghul. Raised by the League of Assassins, he became the fifth Robin.", "son, Talia"),
  first="Damian Wayne first appeared as a named character in Batman #655 in 2006.",
  more=[(["Who is Damian Wayne's mother?", "Who is Damian's mom?", "Who are Damian Wayne's parents?"],
         "Damian Wayne's mother is Talia al Ghul and his father is Bruce Wayne.", "Talia")])

E("Stephanie Brown", "spoiler|the spoiler",
  who=("Stephanie Brown is the vigilante Spoiler, daughter of the villain Cluemaster. She served briefly as the fourth Robin and later as Batgirl.", "Spoiler"),
  first="Stephanie Brown first appeared in Detective Comics #647 in 1992.")

E("Barbara Gordon", "oracle|babs",
  who=("Barbara Gordon is the best-known Batgirl and the daughter of Commissioner James Gordon. "
       "After the Joker shot and paralyzed her in The Killing Joke, she became the computer expert Oracle.", "Batgirl"),
  first="Barbara Gordon first appeared in Detective Comics #359 in 1967.",
  more=[(["Who is Oracle?", "What is Oracle's real name?", "Which Batgirl became Oracle?"],
         "Oracle is Barbara Gordon, the former Batgirl.", "Barbara Gordon"),
        (["Who paralyzed Barbara Gordon?", "Who shot Batgirl?", "How did Barbara Gordon end up in a wheelchair?"],
         "The Joker shot and paralyzed Barbara Gordon in The Killing Joke.", "Joker")])

E("Batgirl", "bat-girl",
  who=("Batgirl is a heroine who works alongside Batman. The best-known Batgirl is Barbara Gordon; Cassandra Cain and Stephanie Brown have also held the name.", "Barbara Gordon"),
  real=("Batgirl's real name is usually Barbara Gordon.", "Barbara Gordon"))

E("Cassandra Cain", "orphan|black bat",
  who=("Cassandra Cain is a Batgirl and one of the best fighters in the DC Universe. "
       "She is the daughter of the assassins David Cain and Lady Shiva and was raised to read body language instead of speech.", "Batgirl"),
  first="Cassandra Cain first appeared in Batman #567 in 1999.")

E("Batwoman", "kate kane|katherine kane",
  who=("Batwoman is Kate Kane, a wealthy former military cadet and a cousin of Bruce Wayne who fights crime in Gotham City.", "Kate Kane|Kane"),
  real=("Batwoman's real name is Kate Kane.", "Kate Kane|Katherine Kane"),
  first="The modern Batwoman, Kate Kane, first appeared in 52 #7 in 2006.")

E("James Gordon", "commissioner gordon|jim gordon|gordon", page="Commissioner Gordon",
  who=("James Gordon is the commissioner of the Gotham City Police Department and one of Batman's most trusted allies. He is the father of Barbara Gordon.", "commissioner|police"),
  first="James Gordon first appeared in Detective Comics #27 in 1939.",
  by="James Gordon was created by Bill Finger and Bob Kane.")

E("Lucius Fox", "",
  who=("Lucius Fox is the businessman who runs Wayne Enterprises for Bruce Wayne. In many versions he also supplies Batman's equipment.", "Wayne Enterprises"),
  first="Lucius Fox first appeared in Batman #307 in 1979.")

E("Catwoman", "selina kyle|the cat",
  who=("Catwoman is Selina Kyle, a skilled cat burglar in Gotham City. She is sometimes Batman's enemy, sometimes his ally, and his most enduring love interest.", "Selina Kyle"),
  real=("Catwoman's real name is Selina Kyle.", "Selina Kyle"),
  first="Catwoman first appeared in Batman #1 in 1940, where she was called the Cat.",
  by="Catwoman was created by Bill Finger and Bob Kane.")

E("Harvey Bullock", "detective bullock",
  who=("Harvey Bullock is a gruff detective in the Gotham City Police Department and a loyal colleague of Commissioner Gordon.", "detective"),
  first="Harvey Bullock first appeared in Detective Comics #441 in 1974.")

E("Renee Montoya", "",
  who=("Renee Montoya is a Gotham City police detective who was created for Batman: The Animated Series. In the comics she later became the faceless vigilante the Question.", "detective|Question"))

E("Leslie Thompkins", "dr. thompkins",
  who=("Leslie Thompkins is a Gotham doctor and a friend of the Wayne family who comforted young Bruce after his parents' murder. She runs a free clinic and knows he is Batman.", "doctor|physician"))

E("Vicki Vale", "",
  who=("Vicki Vale is a Gotham City journalist and photographer and one of Bruce Wayne's love interests.", "journalist|reporter|photo"),
  first="Vicki Vale first appeared in Batman #49 in 1948.")

E("Azrael", "jean-paul valley", page="Azrael",
  who=("Azrael is Jean-Paul Valley, an assassin trained by the Order of St. Dumas. He replaced Bruce Wayne as Batman during Knightfall after Bane broke Batman's back.", "Jean-Paul Valley"),
  real=("Azrael's real name is Jean-Paul Valley.", "Jean-Paul Valley"),
  first="Azrael first appeared in Batman: Sword of Azrael #1 in 1992.")

E("the Huntress", "huntress|helena bertinelli", page="Huntress (Helena Bertinelli)",
  who=("The Huntress is Helena Bertinelli, the daughter of a murdered Gotham mob family who hunts criminals with a crossbow.", "Helena Bertinelli"),
  real=("The Huntress's real name is Helena Bertinelli.", "Helena Bertinelli"))

E("Batwing", "luke fox|david zavimbe",
  who=("Batwing is an armored hero in Batman's circle. The first Batwing was David Zavimbe, a police officer in Africa; the second is Luke Fox, the son of Lucius Fox.", "Luke Fox|Zavimbe"))

E("Duke Thomas", "the signal",
  who=("Duke Thomas is the Signal, a young Gotham hero and ally of Batman who works in daylight.", "Signal"))

E("Ace the Bat-Hound", "ace|bat-hound|batman's dog",
  who=("Ace the Bat-Hound is Batman's dog, a German Shepherd who helps him fight crime.", "dog"),
  first="Ace the Bat-Hound first appeared in Batman #92 in 1955.", kind="what")

E("Bat-Mite", "batmite",
  who=("Bat-Mite is a small imp from another dimension with magical powers. He idolizes Batman and causes trouble by trying to help him.", "imp"),
  first="Bat-Mite first appeared in Detective Comics #267 in 1959.")

E("Talia al Ghul", "talia",
  who=("Talia al Ghul is the daughter of Ra's al Ghul, a love interest of Batman and the mother of his son, Damian Wayne.", "Ra's al Ghul, Damian"),
  first="Talia al Ghul first appeared in Detective Comics #411 in 1971.")

E("Terry McGinnis", "batman beyond batman", page="Batman (Terry McGinnis)",
  who=("Terry McGinnis is the Batman of the future in Batman Beyond, a teenager trained by an elderly Bruce Wayne.", "Batman Beyond|future"))

E("Carrie Kelley", "",
  who=("Carrie Kelley is the teenage girl who becomes Robin in Frank Miller's The Dark Knight Returns.", "Robin, Dark Knight Returns"))

E("the Gotham City Police Department", "gcpd|gotham police", kind="what", page="Gotham City Police Department",
  who=("The Gotham City Police Department, or GCPD, is Gotham's police force. It is led by Commissioner James Gordon and calls Batman with the Bat-Signal.", "police"))

# ------------------------------------------------------------------ villains
E("the Joker", "joker|the clown prince of crime|clown prince of crime", page="Joker",
  who=("The Joker is Batman's archenemy, a criminal mastermind with chalk-white skin, green hair and a permanent grin. "
       "He has no superpowers and commits crimes for his own amusement.", "archenemy|arch-enemy|enemy"),
  real=("The Joker has no definitive real name in the main comics, and his past is deliberately left uncertain. "
        "On screen he is Jack Napier in the 1989 film Batman and Arthur Fleck in the 2019 film Joker.", "no definitive|unknown|never|uncertain, Napier|Fleck"),
  first="The Joker first appeared in Batman #1 in 1940.",
  by="The Joker was created by Bill Finger, Bob Kane and Jerry Robinson.",
  more=[(["What is the Joker's origin?", "How did the Joker become the Joker?", "Where did the Joker come from?", "What happened to make the Joker?"],
         "The best-known origin is that he fell into a vat of chemicals that bleached his skin and drove him insane. The Joker himself says he remembers his past differently each time.", "chemical"),
        (["Who is the Joker's girlfriend?", "Who is the Joker's partner in crime?", "Who is in love with the Joker?"],
         "The Joker's best-known partner is Harley Quinn.", "Harley Quinn")])

E("Harley Quinn", "harleen quinzel|dr. harleen quinzel|harley",
  who=("Harley Quinn is Dr. Harleen Quinzel, a psychiatrist at Arkham Asylum who fell in love with her patient, the Joker, and became his accomplice. "
       "She later left him and became an antihero.", "Harleen Quinzel|psychiatrist"),
  real=("Harley Quinn's real name is Dr. Harleen Quinzel.", "Harleen Quinzel"),
  first="Harley Quinn first appeared in the Batman: The Animated Series episode \"Joker's Favor\" in 1992.",
  by="Harley Quinn was created by Paul Dini and Bruce Timm.")

E("the Penguin", "penguin|oswald cobblepot|cobblepot", page="Penguin",
  who=("The Penguin is Oswald Cobblepot, a Gotham City crime boss known for his short stature, top hat and trick umbrellas. He runs the Iceberg Lounge nightclub.", "Oswald Cobblepot|Cobblepot"),
  real=("The Penguin's real name is Oswald Chesterfield Cobblepot.", "Cobblepot"),
  first="The Penguin first appeared in Detective Comics #58 in 1941.",
  by="The Penguin was created by Bob Kane and Bill Finger.")

E("the Riddler", "riddler|edward nygma|edward nigma|e. nygma", page="Riddler",
  who=("The Riddler is Edward Nygma, a criminal genius who is compelled to leave riddles and puzzles as clues to his crimes.", "Nygma|Nigma|riddle"),
  real=("The Riddler's real name is Edward Nygma, also spelled Nigma.", "Nygma|Nigma"),
  first="The Riddler first appeared in Detective Comics #140 in 1948.",
  by="The Riddler was created by Bill Finger and Dick Sprang.")

E("Two-Face", "harvey dent|two face",
  who=("Two-Face is Harvey Dent, Gotham City's former district attorney. After acid scarred half his face he became a criminal who makes decisions by flipping a coin.", "Harvey Dent"),
  real=("Two-Face's real name is Harvey Dent.", "Harvey Dent"),
  first="Two-Face first appeared in Detective Comics #66 in 1942.",
  by="Two-Face was created by Bob Kane and Bill Finger.",
  more=[(["Who scarred Harvey Dent's face?", "How did Two-Face get his scars?", "Who threw acid at Harvey Dent?"],
         "The mob boss Sal Maroni threw acid in Harvey Dent's face during a trial.", "Maroni")])

E("Bane", "", page="Bane",
  who=("Bane is a brilliant strategist and one of Batman's strongest enemies. He uses the drug Venom to boost his strength and is known as the man who broke Batman's back.", "Venom|back"),
  first="Bane first appeared in Batman: Vengeance of Bane #1 in 1993.",
  by="Bane was created by Chuck Dixon, Doug Moench and Graham Nolan.",
  more=[(["Who broke Batman's back?", "Which villain broke the Bat?", "Who crippled Batman in Knightfall?", "Who snapped Batman's spine?"],
         "Bane broke Batman's back in the Knightfall storyline.", "Bane")])

E("the Scarecrow", "scarecrow|jonathan crane|dr. crane", page="Scarecrow",
  who=("The Scarecrow is Dr. Jonathan Crane, a former psychology professor who uses a fear toxin to make his victims see their worst fears.", "Jonathan Crane|fear"),
  real=("The Scarecrow's real name is Dr. Jonathan Crane.", "Crane"),
  first="The Scarecrow first appeared in World's Finest Comics #3 in 1941.")

E("Ra's al Ghul", "ras al ghul|the demon's head|the demon",
  who=("Ra's al Ghul is the centuries-old leader of the League of Assassins, who stays alive by bathing in the Lazarus Pits. "
       "He is the father of Talia al Ghul and wants Batman as his heir.", "League of Assassins|Lazarus"),
  first="Ra's al Ghul first appeared in Batman #232 in 1971.",
  by="Ra's al Ghul was created by Dennis O'Neil, Neal Adams and Julius Schwartz.",
  more=[(["What does Ra's al Ghul mean?", "What does the name Ra's al Ghul translate to?", "What is the meaning of Ra's al Ghul's name?"],
         "Ra's al Ghul is Arabic for \"the Demon's Head\".", "Demon")])

E("Mr. Freeze", "mister freeze|victor fries|dr. victor fries|mr freeze",
  who=("Mr. Freeze is Dr. Victor Fries, a scientist who must wear a refrigerated suit to survive and who fights with a freeze gun. "
       "He turns to crime to save his terminally ill wife, Nora.", "Victor Fries|Fries"),
  real=("Mr. Freeze's real name is Dr. Victor Fries.", "Fries"),
  first="Mr. Freeze first appeared in Batman #121 in 1959, under the name Mr. Zero.")

E("Poison Ivy", "pamela isley|dr. pamela isley|ivy",
  who=("Poison Ivy is Dr. Pamela Isley, a botanist who can control plants and whose touch is poisonous. She commits crimes to protect the natural world.", "Pamela Isley|plants"),
  real=("Poison Ivy's real name is Dr. Pamela Isley.", "Isley"),
  first="Poison Ivy first appeared in Batman #181 in 1966.")

E("Killer Croc", "waylon jones|croc",
  who=("Killer Croc is Waylon Jones, a criminal with a rare condition that gives him crocodile-like skin, great strength and sharp teeth.", "Waylon Jones|crocodile"),
  real=("Killer Croc's real name is Waylon Jones.", "Waylon Jones"))

E("Clayface", "basil karlo|matt hagen",
  who=("Clayface is a name used by several Batman villains whose clay-like bodies let them change shape. The first was Basil Karlo, a horror-film actor.", "Basil Karlo|clay|shape"),
  real=("The original Clayface's real name is Basil Karlo.", "Basil Karlo"),
  first="Clayface first appeared in Detective Comics #40 in 1940.")

E("the Mad Hatter", "mad hatter|jervis tetch", page="Mad Hatter",
  who=("The Mad Hatter is Jervis Tetch, a scientist obsessed with Alice in Wonderland who controls people's minds with devices hidden in hats.", "Jervis Tetch|mind"),
  real=("The Mad Hatter's real name is Jervis Tetch.", "Tetch"),
  first="The Mad Hatter first appeared in Batman #49 in 1948.")

E("Hush", "thomas elliot|tommy elliot", page="Hush",
  who=("Hush is Dr. Thomas Elliot, a surgeon and childhood friend of Bruce Wayne who hides his face in bandages and sets out to destroy him.", "Thomas Elliot|Elliot"),
  real=("Hush's real name is Dr. Thomas Elliot.", "Elliot"),
  first="Hush first appeared in Batman #609 in 2003.",
  by="Hush was created by Jeph Loeb and Jim Lee.")

E("Black Mask", "roman sionis", page="Black Mask",
  who=("Black Mask is Roman Sionis, a ruthless Gotham City crime lord who wears a black skull mask.", "Roman Sionis|crime"),
  real=("Black Mask's real name is Roman Sionis.", "Sionis"),
  first="Black Mask first appeared in Batman #386 in 1985.")

E("Deathstroke", "slade wilson|deathstroke the terminator|slade",
  who=("Deathstroke is Slade Wilson, a mercenary and assassin with enhanced strength and reflexes. He began as an enemy of the Teen Titans and often fights Batman.", "Slade Wilson|mercenary|assassin"),
  real=("Deathstroke's real name is Slade Wilson.", "Slade Wilson"),
  first="Deathstroke first appeared in The New Teen Titans #2 in 1980.",
  by="Deathstroke was created by Marv Wolfman and George Pérez.")

E("the Court of Owls", "court of owls|the talons|talons", kind="what", page="Court of Owls",
  who=("The Court of Owls is a secret society of wealthy families that has controlled Gotham City for centuries. Its assassins are called Talons.", "secret|Talon"),
  by="The Court of Owls was created by Scott Snyder and Greg Capullo.")

E("Man-Bat", "kirk langstrom|dr. kirk langstrom|man bat",
  who=("Man-Bat is Dr. Kirk Langstrom, a scientist whose serum made from bat DNA turns him into a giant bat creature.", "Kirk Langstrom|Langstrom"),
  real=("Man-Bat's real name is Dr. Kirk Langstrom.", "Langstrom"),
  first="Man-Bat first appeared in Detective Comics #400 in 1970.")

E("Hugo Strange", "professor hugo strange|dr. hugo strange",
  who=("Hugo Strange is a brilliant, corrupt psychiatrist and one of Batman's earliest enemies. He is one of the few villains to work out that Batman is Bruce Wayne.", "psychiatrist|psychologist"),
  first="Hugo Strange first appeared in Detective Comics #36 in 1940.")

E("Victor Zsasz", "zsasz|mr. zsasz",
  who=("Victor Zsasz is a serial killer who carves a tally mark into his own skin for every person he murders.", "serial killer|tally"),
  first="Victor Zsasz first appeared in Batman: Shadow of the Bat #1 in 1992.")

E("Carmine Falcone", "falcone|the roman",
  who=("Carmine Falcone, nicknamed the Roman, is a mafia boss who ruled Gotham City's organized crime in Batman's early years.", "mafia|mob|crime"))

E("Sal Maroni", "salvatore maroni|maroni",
  who=("Sal Maroni is a Gotham City mob boss, best known for throwing acid at district attorney Harvey Dent and turning him into Two-Face.", "acid|Dent|Two-Face"))

E("Firefly", "garfield lynns", page="Firefly",
  who=("Firefly is Garfield Lynns, a pyromaniac arsonist who wears a flying suit and carries a flamethrower.", "Garfield Lynns|pyromaniac|arson|fire"),
  real=("Firefly's real name is Garfield Lynns.", "Lynns"))

E("Calendar Man", "julian day",
  who=("Calendar Man is Julian Day, a criminal who plans his crimes around holidays and significant dates.", "Julian Day|holiday|dates"),
  real=("Calendar Man's real name is Julian Day.", "Julian Day"))

E("Professor Pyg", "lazlo valentin|pyg",
  who=("Professor Pyg is Lazlo Valentin, a deranged surgeon in a pig mask who turns his victims into mindless servants called Dollotrons.", "Lazlo Valentin|surgeon|Dollotron"),
  by="Professor Pyg was created by Grant Morrison and Andy Kubert.")

E("Deadshot", "floyd lawton",
  who=("Deadshot is Floyd Lawton, an expert marksman and hired assassin who claims he never misses. He is often a member of the Suicide Squad.", "Floyd Lawton|marksman|assassin"),
  real=("Deadshot's real name is Floyd Lawton.", "Lawton"),
  first="Deadshot first appeared in Batman #59 in 1950.")

E("Solomon Grundy", "cyrus gold|grundy", page="Solomon Grundy",
  who=("Solomon Grundy is a huge, super-strong zombie who rose from Slaughter Swamp near Gotham City. In life he was a merchant named Cyrus Gold.", "zombie|undead|Cyrus Gold"))

E("Lady Shiva", "sandra wu-san|shiva",
  who=("Lady Shiva is one of the deadliest martial artists in the DC Universe, an assassin who is the mother of Cassandra Cain.", "martial artist|assassin|Cassandra"))

E("the League of Assassins", "league of assassins|league of shadows|the league of shadows", kind="what", page="League of Assassins",
  who=("The League of Assassins is a secret order of killers led by Ra's al Ghul. In the Dark Knight film trilogy it is called the League of Shadows.", "Ra's al Ghul"))

E("the Batman Who Laughs", "batman who laughs", page="The Batman Who Laughs",
  who=("The Batman Who Laughs is an evil Bruce Wayne from a dark alternate Earth who was infected by the Joker's toxin, giving him Batman's mind and the Joker's cruelty.", "alternate|Joker"),
  by="The Batman Who Laughs was created by Scott Snyder and Greg Capullo.")

E("Punchline", "alexis kaye", page="Punchline",
  who=("Punchline is Alexis Kaye, a college student who became the Joker's girlfriend and partner after Harley Quinn left him.", "Alexis Kaye|Joker"),
  first="Punchline first appeared in Batman #89 in 2020.")

E("the Ventriloquist", "ventriloquist|arnold wesker|scarface", page="Ventriloquist",
  who=("The Ventriloquist is Arnold Wesker, a meek man who commits crimes through his gangster puppet, Scarface.", "Arnold Wesker|puppet|Scarface|dummy"))

E("Killer Moth", "drury walker",
  who=("Killer Moth is Drury Walker, a criminal who set himself up as an anti-Batman, hiring himself out to protect other crooks.", "Drury Walker|criminal"))

E("Anarky", "lonnie machin",
  who=("Anarky is Lonnie Machin, a teenage prodigy who attacks corrupt institutions in the name of anarchist ideas.", "Lonnie Machin|anarch"),
  by="Anarky was created by Alan Grant and Norm Breyfogle.")

E("KGBeast", "anatoli knyazev",
  who=("KGBeast is Anatoli Knyazev, a Soviet-trained assassin with a weapon in place of the hand he cut off to escape Batman.", "Anatoli Knyazev|assassin"))

E("Kite Man", "charles brown|chuck brown",
  who=("Kite Man is Charles \"Chuck\" Brown, a minor villain who commits crimes with kites and a hang glider. His catchphrase is \"Hell yeah\".", "kite"))

E("Cluemaster", "arthur brown",
  who=("Cluemaster is Arthur Brown, a failed game-show host who leaves clues to his crimes. He is the father of Stephanie Brown, the hero Spoiler.", "Arthur Brown|Stephanie|clue"))

E("Tony Zucco", "zucco",
  who=("Tony Zucco is the gangster who killed Dick Grayson's parents by sabotaging their trapeze act, which led Dick to become Robin.", "Grayson|parents"))

E("Owlman", "", page="Owlman",
  who=("Owlman is an evil counterpart of Batman from an alternate Earth, a member of the Crime Syndicate.", "alternate|Crime Syndicate|counterpart"))

E("the Arkham Knight", "arkham knight", page="Batman: Arkham Knight",
  who=("The Arkham Knight is the masked military commander who hunts Batman in the 2015 video game Batman: Arkham Knight. He is revealed to be Jason Todd.", "Jason Todd"))

# ------------------------------------------------------------------ places and gear
E("Gotham City", "gotham", kind="what",
  who=("Gotham City is the dark, crime-ridden American city that Batman protects. It is home to Wayne Manor, Arkham Asylum and the Gotham City Police Department.", "city"),
  more=[(["What city does Batman protect?", "Where does Batman live?", "Which city is Batman from?", "Where does Batman fight crime?"],
         "Batman protects Gotham City.", "Gotham")])

E("Arkham Asylum", "arkham|the asylum", kind="what",
  who=("Arkham Asylum is the psychiatric hospital in Gotham City where many of Batman's enemies, including the Joker, are locked up. Its inmates often escape.", "psychiatric|hospital|asylum"),
  first="Arkham Asylum first appeared in Batman #258 in 1974.",
  more=[(["Where are Batman's villains locked up?", "Where is the Joker imprisoned?", "Which hospital holds Gotham's criminally insane?"],
         "Batman's most dangerous enemies are held in Arkham Asylum.", "Arkham")])

E("the Batcave", "batcave|bat cave", kind="what", page="Batcave",
  who=("The Batcave is Batman's secret headquarters, a network of caves beneath Wayne Manor. It holds the Batcomputer, his vehicles, a crime lab and a trophy room.", "Wayne Manor|headquarters"),
  more=[(["Where is the Batcave?", "Where is Batman's headquarters?", "Where is Batman's secret base?"],
         "The Batcave is underneath Wayne Manor.", "Wayne Manor")])

E("Wayne Manor", "", kind="what",
  who=("Wayne Manor is the Wayne family mansion on the outskirts of Gotham City, where Bruce Wayne lives. The Batcave lies beneath it.", "mansion|Batcave|home"))

E("Wayne Enterprises", "waynetech|wayne industries", kind="what",
  who=("Wayne Enterprises is the business empire Bruce Wayne inherited from his family. Its wealth and technology pay for Batman's equipment.", "Bruce Wayne|company|business"))

E("the Batmobile", "batmobile|batman's car", kind="what", page="Batmobile",
  who=("The Batmobile is Batman's car, a heavily armored vehicle packed with gadgets and weapons.", "car|vehicle"),
  more=[(["What car does Batman drive?", "What is Batman's car called?", "What does Batman drive?"],
         "Batman drives the Batmobile.", "Batmobile")])

E("the Batsuit", "batsuit|bat suit|batman's costume|batman's suit", kind="what", page="Batsuit",
  who=("The Batsuit is Batman's costume: a gray and black armored suit with a cape, a cowl with pointed ears and a bat symbol on the chest. It protects him and frightens criminals.", "costume|cape|cowl|suit"))

E("the utility belt", "utility belt|batman's belt", kind="what", page="Batman's utility belt",
  who=("The utility belt is the belt Batman wears to carry his equipment, such as Batarangs, a grapple gun, smoke pellets and forensic tools.", "equipment|Batarang|gadgets|tools"))

E("the Bat-Signal", "bat-signal|bat signal|batsignal", kind="what", page="Bat-Signal",
  who=("The Bat-Signal is a searchlight that projects a bat symbol onto the sky over Gotham City. The police use it to call Batman.", "searchlight|light|signal"),
  more=[(["How do the police call Batman?", "How does Commissioner Gordon contact Batman?", "How does Gotham summon Batman?"],
         "The police call Batman by shining the Bat-Signal into the sky.", "Bat-Signal")])

E("the Batarang", "batarang|batarangs", kind="what", page="Batarang",
  who=("The Batarang is a bat-shaped throwing weapon that Batman carries in his utility belt.", "throwing|weapon"))

E("the Batplane", "batplane|batwing jet|batman's plane", kind="what", page="Batplane",
  who=("The Batplane, also called the Batwing, is Batman's aircraft.", "aircraft|plane|jet"))

E("the Batcomputer", "batcomputer|bat computer", kind="what", page="Batcomputer",
  who=("The Batcomputer is the supercomputer in the Batcave that Batman uses to analyze evidence and track criminals.", "computer"))

E("Blackgate Penitentiary", "blackgate|blackgate prison", kind="what",
  who=("Blackgate Penitentiary is Gotham City's prison for criminals who are not insane, unlike those sent to Arkham Asylum.", "prison"))

E("the Lazarus Pit", "lazarus pit|lazarus pits", kind="what", page="Lazarus Pit",
  who=("A Lazarus Pit is a pool of chemicals that can heal injuries and bring back the dead. Ra's al Ghul uses the pits to live for centuries.", "Ra's al Ghul|heal|dead|restor"))

E("the Iceberg Lounge", "iceberg lounge", kind="what", page="Iceberg Lounge",
  who=("The Iceberg Lounge is the Gotham City nightclub owned by the Penguin, which he uses as a front for his criminal business.", "Penguin|nightclub"))

E("Venom", "venom drug", kind="what", page="Bane",
  who=("Venom is the super-steroid that gives Bane his enormous strength. It is pumped into his body through tubes.", "Bane|strength|steroid|drug"))

# ------------------------------------------------------------------ comics
E("Detective Comics", "", kind="what",
  who=("Detective Comics is the comic book series in which Batman first appeared, in issue #27 in 1939. The publisher DC Comics takes its name from it.", "27|DC"))

E("Batman: Year One", "year one", kind="what",
  who=("Batman: Year One is a 1987 story by writer Frank Miller and artist David Mazzucchelli that retells Bruce Wayne's first year as Batman and James Gordon's arrival in Gotham.", "Miller|first year"),
  more=[(["Who wrote Batman: Year One?", "Who is the author of Year One?", "Which writer created Batman: Year One?"],
         "Batman: Year One was written by Frank Miller and drawn by David Mazzucchelli.", "Miller")])

E("The Killing Joke", "batman: the killing joke|killing joke", kind="what", page="Batman: The Killing Joke",
  who=("The Killing Joke is a 1988 graphic novel by writer Alan Moore and artist Brian Bolland. The Joker shoots and paralyzes Barbara Gordon to try to drive her father insane, and the book gives a possible origin for the Joker.", "Moore|Joker"),
  more=[(["Who wrote The Killing Joke?", "Who is the author of The Killing Joke?", "Which writer is behind The Killing Joke?"],
         "The Killing Joke was written by Alan Moore and drawn by Brian Bolland.", "Moore")])

E("The Dark Knight Returns", "dark knight returns|tdkr", kind="what",
  who=("The Dark Knight Returns is a 1986 comic by Frank Miller in which a 55-year-old Bruce Wayne comes out of retirement to be Batman again.", "Miller|retire"),
  more=[(["Who wrote The Dark Knight Returns?", "Who created The Dark Knight Returns?", "Which writer made The Dark Knight Returns?"],
         "The Dark Knight Returns was written and drawn by Frank Miller.", "Miller")])

E("Knightfall", "batman: knightfall", kind="what", page="Batman: Knightfall",
  who=("Knightfall is a 1993 to 1994 storyline in which Bane breaks Batman's back and Jean-Paul Valley, Azrael, takes over as a more violent Batman until Bruce Wayne returns.", "Bane"))

E("The Long Halloween", "batman: the long halloween|long halloween", kind="what", page="Batman: The Long Halloween",
  who=("The Long Halloween is a 1996 to 1997 series by writer Jeph Loeb and artist Tim Sale. Batman hunts Holiday, a killer who strikes on holidays, and Harvey Dent becomes Two-Face.", "Loeb|Holiday"),
  more=[(["Who wrote The Long Halloween?", "Who is the author of The Long Halloween?", "Which writer created The Long Halloween?"],
         "The Long Halloween was written by Jeph Loeb and drawn by Tim Sale.", "Loeb")])

E("Batman: Hush", "hush storyline|hush comic", kind="what",
  who=("Batman: Hush is a 2002 to 2003 storyline by writer Jeph Loeb and artist Jim Lee in which a mysterious bandaged enemy turns Batman's foes against him.", "Loeb|Lee"))

E("A Death in the Family", "batman: a death in the family|death in the family", kind="what", page="A Death in the Family",
  who=("A Death in the Family is a 1988 storyline in which the Joker kills Jason Todd, the second Robin. Readers decided his fate by telephone vote.", "Jason Todd"))

E("No Man's Land", "batman: no man's land", kind="what", page="Batman: No Man's Land",
  who=("No Man's Land is a 1999 storyline in which Gotham City is wrecked by an earthquake, abandoned by the United States government and carved up by gangs and villains.", "earthquake"))

E("Under the Hood", "batman: under the hood|under the red hood", kind="what", page="Batman: Under the Hood",
  who=("Under the Hood is a 2005 storyline by writer Judd Winick in which Jason Todd, the Robin killed by the Joker, returns as the Red Hood.", "Jason Todd|Red Hood"))

E("Batman R.I.P.", "batman rip", kind="what",
  who=("Batman R.I.P. is a 2008 storyline by writer Grant Morrison in which the Black Glove, led by Dr. Simon Hurt, tries to destroy Batman's mind.", "Morrison"))

E("Batman: Three Jokers", "three jokers", kind="what",
  who=("Batman: Three Jokers is a 2020 series by writer Geoff Johns and artist Jason Fabok built on the idea that three different men have been the Joker.", "Johns|three"))

E("Arkham Asylum: A Serious House on Serious Earth", "a serious house on serious earth", kind="what",
  who=("Arkham Asylum: A Serious House on Serious Earth is a 1989 graphic novel by writer Grant Morrison and artist Dave McKean in which Batman enters the asylum after its inmates take it over.", "Morrison|McKean"))

# ------------------------------------------------------------------ film, television and games
E("the 1989 film Batman", "batman 1989|the 1989 batman film|tim burton's batman|batman (1989 film)", kind="what", title="Batman (1989 film)",
  who=("Batman is a 1989 film directed by Tim Burton, starring Michael Keaton as Batman and Jack Nicholson as the Joker.", "Burton|Keaton"))

E("Batman Returns", "", kind="what",
  who=("Batman Returns is a 1992 film directed by Tim Burton, with Michael Keaton as Batman, Danny DeVito as the Penguin and Michelle Pfeiffer as Catwoman.", "Burton|Keaton"))

E("Batman Forever", "", kind="what",
  who=("Batman Forever is a 1995 film directed by Joel Schumacher, with Val Kilmer as Batman, Tommy Lee Jones as Two-Face and Jim Carrey as the Riddler.", "Schumacher|Kilmer"))

E("the 1997 film Batman & Robin", "batman and robin film|batman & robin|batman and robin movie", kind="what", title="Batman & Robin (film)",
  who=("Batman & Robin is a 1997 film directed by Joel Schumacher, with George Clooney as Batman, Arnold Schwarzenegger as Mr. Freeze and Uma Thurman as Poison Ivy.", "Schumacher|Clooney"))

E("Batman Begins", "", kind="what",
  who=("Batman Begins is a 2005 film directed by Christopher Nolan, starring Christian Bale as Batman. It tells his origin and pits him against Ra's al Ghul and the Scarecrow.", "Nolan|Bale"))

E("The Dark Knight", "the dark knight film|dark knight movie", kind="what",
  who=("The Dark Knight is a 2008 film directed by Christopher Nolan, with Christian Bale as Batman and Heath Ledger as the Joker. Ledger won a posthumous Academy Award for the role.", "Nolan|Ledger"),
  more=[(["Who played the Joker in The Dark Knight?", "Which actor was the Joker in The Dark Knight?", "Who was the Joker in the 2008 film?"],
         "Heath Ledger played the Joker in The Dark Knight.", "Ledger")])

E("The Dark Knight Rises", "", kind="what",
  who=("The Dark Knight Rises is a 2012 film directed by Christopher Nolan, the last of his trilogy, with Christian Bale as Batman, Tom Hardy as Bane and Anne Hathaway as Selina Kyle.", "Nolan|Hardy|Bane"))

E("the 2022 film The Batman", "the batman 2022|the batman movie|the batman film|the batman (2022 film)", kind="what", title="The Batman (film)",
  who=("The Batman is a 2022 film directed by Matt Reeves, starring Robert Pattinson as Batman, with Paul Dano as the Riddler, Zoë Kravitz as Selina Kyle and Colin Farrell as the Penguin.", "Reeves|Pattinson"))

E("the 2019 film Joker", "joker movie|joker film|joker 2019|joker (2019 film)", kind="what", title="Joker (2019 film)",
  who=("Joker is a 2019 film directed by Todd Phillips, starring Joaquin Phoenix as Arthur Fleck, a failed comedian who becomes the Joker. Phoenix won the Academy Award for Best Actor.", "Phillips|Phoenix"))

E("the 1966 Batman TV series", "batman 1966|batman '66|the adam west batman|1960s batman show", kind="what", page="Batman (TV series)",
  who=("Batman is a television series that ran from 1966 to 1968, starring Adam West as Batman and Burt Ward as Robin. It is known for its campy, comic tone.", "Adam West|West"))

E("Batman: The Animated Series", "the animated series|btas", kind="what",
  who=("Batman: The Animated Series is a television cartoon that ran from 1992 to 1995, developed by Bruce Timm and Eric Radomski. Kevin Conroy voiced Batman and Mark Hamill voiced the Joker, and it introduced Harley Quinn.", "Timm|Conroy|1992"),
  more=[(["Who voiced Batman in The Animated Series?", "Who was the voice of Batman in the 1990s cartoon?", "Which actor voiced the animated Batman?"],
         "Kevin Conroy voiced Batman in Batman: The Animated Series.", "Conroy"),
        (["Who voiced the Joker in The Animated Series?", "Who is the voice of the animated Joker?", "Which actor voiced the Joker in the 1990s cartoon?"],
         "Mark Hamill voiced the Joker in Batman: The Animated Series.", "Hamill")])

E("Batman Beyond", "", kind="what",
  who=("Batman Beyond is an animated series that ran from 1999 to 2001. In a future Gotham, an elderly Bruce Wayne trains teenager Terry McGinnis as the new Batman.", "Terry McGinnis|future"))

E("the television series Gotham", "gotham tv show|gotham series|gotham tv series|gotham show", kind="what", title="Gotham (TV series)",
  who=("Gotham is a television series that ran from 2014 to 2019 about a young James Gordon and a teenage Bruce Wayne in the years before Batman.", "Gordon"))

E("the television series The Penguin", "the penguin show|the penguin series|the penguin tv series", kind="what", title="The Penguin (TV series)",
  who=("The Penguin is a 2024 television series, a spin-off of the film The Batman, starring Colin Farrell as Oz Cobb.", "Farrell"))

E("The Lego Batman Movie", "lego batman", kind="what",
  who=("The Lego Batman Movie is a 2017 animated comedy film in which Will Arnett voices Batman.", "Arnett|2017"))

E("Batman: Arkham Asylum", "arkham asylum game", kind="what",
  who=("Batman: Arkham Asylum is a 2009 video game developed by Rocksteady Studios in which Batman is trapped in the asylum after the Joker takes it over. It began the Batman: Arkham series.", "Rocksteady|2009"))

E("Batman: Arkham City", "arkham city", kind="what",
  who=("Batman: Arkham City is a 2011 video game by Rocksteady Studios, the sequel to Arkham Asylum, set in a walled-off prison district of Gotham City.", "Rocksteady|2011"))

E("Batman: Arkham Knight", "arkham knight game", kind="what",
  who=("Batman: Arkham Knight is a 2015 video game by Rocksteady Studios in which Batman faces the Scarecrow and the mysterious Arkham Knight.", "Rocksteady|2015"))

E("Batman: Mask of the Phantasm", "mask of the phantasm", kind="what",
  who=("Batman: Mask of the Phantasm is a 1993 animated film based on Batman: The Animated Series, in which Batman is blamed for murders committed by a masked killer called the Phantasm.", "1993|Phantasm"))

# ------------------------------------------------------------------ people behind Batman
E("Bob Kane", "",
  who=("Bob Kane was the American comic book artist who created Batman with writer Bill Finger in 1939.", "created|creator|artist"))
E("Bill Finger", "",
  who=("Bill Finger was the writer who created Batman with Bob Kane and shaped much of the character, including the costume, the Bruce Wayne identity and Gotham City. DC Comics only began crediting him in 2015.", "created|creator|writer"))
E("Kevin Conroy", "",
  who=("Kevin Conroy was the actor best known as the voice of Batman, starting with Batman: The Animated Series in 1992 and continuing in the Arkham video games.", "voice"))
E("Mark Hamill", "",
  who=("Mark Hamill is the actor who has voiced the Joker since Batman: The Animated Series in 1992, including in the Arkham video games.", "Joker|voice"))
E("Frank Miller", "",
  who=("Frank Miller is the writer and artist behind The Dark Knight Returns and Batman: Year One, which made Batman darker in the 1980s.", "Dark Knight Returns|Year One"))
E("Christopher Nolan", "",
  who=("Christopher Nolan is the director of the Dark Knight trilogy: Batman Begins, The Dark Knight and The Dark Knight Rises.", "director|directed|trilogy"))


# ------------------------------------------------------------------ topics: lists and overviews
def T(title, keywords, answer, questions, keys):
    """A list or overview. `keywords` are extra search words that are indexed but never shown."""
    CARDS.append(dict(n=None, title=title, page=None, kind="topic", aliases=[], keywords=keywords,
                      who=None, real=None, first=None, by=None, more=[(questions, answer, keys)]))


T("Batman's main villains", "top best greatest biggest famous main worst villains enemies foes rogues gallery bad guys ranking five ten",
  "There is no official ranking, but Batman's most prominent villains are:\n1. The Joker, his archenemy.\n2. Two-Face, former district attorney Harvey Dent.\n"
  "3. The Penguin, crime boss Oswald Cobblepot.\n4. The Riddler, Edward Nygma.\n5. Bane, who broke Batman's back.\n"
  "Others include Catwoman, the Scarecrow, Ra's al Ghul, Mr. Freeze, Poison Ivy and Harley Quinn.",
  ["What are the top 5 Batman villains?", "Who are Batman's main enemies?", "top 5 batman villains", "List Batman's greatest villains.",
   "Who are the best Batman villains?", "Name Batman's biggest enemies.", "Who is in Batman's rogues gallery?", "Who are the most famous Batman villains?"],
  "Joker, Two-Face, Penguin")

T("The Robins", "robins list all every how many order sidekicks boy wonder",
  "Five characters have been Robin in the main comics, in this order:\n1. Dick Grayson\n2. Jason Todd\n3. Tim Drake\n4. Stephanie Brown\n5. Damian Wayne",
  ["Who are all the Robins?", "List every Robin in order.", "How many Robins have there been?", "Name all of Batman's Robins.", "Who has been Robin?"],
  "Dick Grayson, Jason Todd, Tim Drake, Damian")

T("The Bat-Family", "bat-family batfamily bat family members team allies helpers partners who helps works with",
  "The Bat-Family is the group of allies who fight beside Batman. Its core members are Alfred Pennyworth, Nightwing (Dick Grayson), Batgirl (Barbara Gordon), "
  "Red Hood (Jason Todd), Red Robin (Tim Drake), Robin (Damian Wayne), Batwoman (Kate Kane) and Commissioner James Gordon.",
  ["Who is in the Bat-Family?", "Who are Batman's allies?", "List the members of the Bat-Family.", "Who helps Batman fight crime?", "Who are Batman's partners?"],
  "Nightwing|Dick Grayson, Batgirl|Barbara Gordon, Alfred")

T("Actors who have played Batman", "actors played portrayed batman live-action film movie cast star who list all every",
  "In live-action films Batman has been played by Michael Keaton (1989 and 1992), Val Kilmer (1995), George Clooney (1997), Christian Bale (2005 to 2012), "
  "Ben Affleck (2016 to 2023) and Robert Pattinson (2022). Adam West played him in the 1966 television series.",
  ["Which actors have played Batman?", "Who has played Batman in the movies?", "List the Batman actors.", "Who played Batman on screen?", "Name every live-action Batman."],
  "Keaton, Bale, Pattinson")

T("Actors who have played the Joker", "actors played portrayed joker live-action film movie cast star who list all every",
  "In live action the Joker has been played by Cesar Romero (1966 series), Jack Nicholson (Batman, 1989), Heath Ledger (The Dark Knight, 2008), "
  "Jared Leto (Suicide Squad, 2016) and Joaquin Phoenix (Joker, 2019). Mark Hamill is his best-known voice actor.",
  ["Which actors have played the Joker?", "Who has played the Joker in the movies?", "List the Joker actors.", "Who played the Joker on screen?", "Name every live-action Joker."],
  "Nicholson, Ledger, Phoenix")

T("Batman films in order", "films movies in order list all every live-action chronological release timeline",
  "The live-action Batman films, in release order:\n1. Batman (1966)\n2. Batman (1989)\n3. Batman Returns (1992)\n4. Batman Forever (1995)\n5. Batman & Robin (1997)\n"
  "6. Batman Begins (2005)\n7. The Dark Knight (2008)\n8. The Dark Knight Rises (2012)\n9. Batman v Superman: Dawn of Justice (2016)\n10. The Batman (2022)",
  ["What are the Batman movies in order?", "List all the Batman films.", "How many Batman movies are there?", "Name the live-action Batman films in release order.", "What Batman movies exist?"],
  "Batman Returns, Batman Begins, The Dark Knight")

T("The Dark Knight trilogy", "dark knight trilogy nolan films three movies order",
  "The Dark Knight trilogy is three films directed by Christopher Nolan, starring Christian Bale as Batman: Batman Begins (2005), The Dark Knight (2008) and The Dark Knight Rises (2012).",
  ["What is the Dark Knight trilogy?", "Which films are in the Dark Knight trilogy?", "What are Christopher Nolan's Batman movies?", "List the Nolan Batman films."],
  "Batman Begins, Dark Knight Rises")

T("The Batman: Arkham games", "arkham games video game series order list all rocksteady",
  "The main Batman: Arkham video games, in release order:\n1. Batman: Arkham Asylum (2009)\n2. Batman: Arkham City (2011)\n3. Batman: Arkham Origins (2013)\n4. Batman: Arkham Knight (2015)",
  ["What are the Batman Arkham games in order?", "List the Arkham video games.", "How many Arkham games are there?", "Which games are in the Batman: Arkham series?"],
  "Arkham Asylum, Arkham City, Arkham Knight")

T("Essential Batman comics", "best essential comics graphic novels stories read start reading recommended must-read top storylines",
  "The Batman stories most often recommended are:\n1. Batman: Year One\n2. The Dark Knight Returns\n3. The Killing Joke\n4. The Long Halloween\n5. Batman: Hush\n6. The Court of Owls",
  ["What are the best Batman comics?", "Which Batman comics should I read first?", "What are the essential Batman stories?", "Recommend some Batman graphic novels.", "Where should I start reading Batman?"],
  "Year One, Dark Knight Returns, Killing Joke")

T("Batman's skills and abilities", "skills abilities powers strengths good at fight fighting detective intelligence training how strong smart",
  "Batman has no superpowers. He is a genius-level detective, a master of many martial arts, an expert escape artist and tactician, "
  "and has the wealth to build advanced equipment. He trained his body and mind to peak human condition.",
  ["What are Batman's abilities?", "What skills does Batman have?", "What is Batman good at?", "How does Batman fight without powers?", "What makes Batman dangerous?"],
  "detective, martial arts")

T("Batman's equipment", "gadgets equipment tools weapons gear vehicles uses carries arsenal technology",
  "Batman's main equipment is the Batsuit, the utility belt, Batarangs and a grapple gun. His vehicles include the Batmobile, the Batplane (or Batwing), the Batcycle and the Batboat.",
  ["What gadgets does Batman use?", "What equipment does Batman have?", "What weapons does Batman carry?", "What vehicles does Batman have?", "List Batman's gear."],
  "Batarang, Batmobile")

T("Batman's love interests", "love interests girlfriend girlfriends romance romantic wife married dating relationship partner lovers",
  "Batman's most enduring love interest is Catwoman, Selina Kyle. Others include Talia al Ghul, the mother of his son Damian, and the journalist Vicki Vale.",
  ["Who is Batman's love interest?", "Who is Batman's girlfriend?", "Who does Batman love?", "Is Batman married?", "Who has Batman dated?"],
  "Catwoman|Selina, Talia")

T("Others who have been Batman", "who else been batman replaced successor took over other batmen wore cowl besides bruce wayne",
  "Besides Bruce Wayne, the main comics have had Jean-Paul Valley (Azrael), Dick Grayson and James Gordon take over as Batman. Terry McGinnis is Batman in the future of Batman Beyond.",
  ["Who else has been Batman?", "Who replaced Bruce Wayne as Batman?", "Has anyone other than Bruce Wayne been Batman?", "Which characters have taken over as Batman?"],
  "Dick Grayson, Jean-Paul Valley|Azrael")

T("The Batgirls", "batgirls list all every how many order",
  "The main Batgirls, in order, are Barbara Gordon, Cassandra Cain and Stephanie Brown. Bette Kane was an earlier Bat-Girl in 1961.",
  ["Who are all the Batgirls?", "List every Batgirl.", "How many Batgirls have there been?", "Who has been Batgirl?"],
  "Barbara Gordon, Cassandra Cain, Stephanie Brown")


# ------------------------------------------------------------------ derived text and checks
def facts(card):
    """All (field, answer, keys) for a card, in card order."""
    out = []
    if card["who"]:
        out.append(("who", card["who"][0], card["who"][1]))
    if card["real"]:
        out.append(("real", card["real"][0], card["real"][1]))
    if card["first"]:
        nums = re.findall(r"#(\d+)", card["first"]) + re.findall(r"\b(?:19|20)\d\d\b", card["first"])
        out.append(("first", card["first"], ", ".join(nums)))
    if card["by"]:
        names = re.findall(r"[A-Z][\w'’.-]+(?: [A-Z][\w'’.-]+)+", card["by"].split(" by ", 1)[1])
        out.append(("by", card["by"], ", ".join(n.split()[-1].rstrip(".") for n in names)))
    for i, (_, a, k) in enumerate(card["more"]):
        out.append((f"more{i}", a, k))
    return out


def text(card):
    """The passage shown to the model: the card's answers, without repeats."""
    seen, out = set(), []
    for _, a, _ in facts(card):
        if a not in seen:
            seen.add(a)
            out.append(a)
    return " ".join(out)


def key_hit(answer: str, keys: str) -> bool:
    """keys: comma-separated requirements, each a |-separated list of alternatives."""
    a = answer.lower()
    return all(any(alt.strip().lower() in a for alt in k.split("|")) for k in keys.split(",") if k.strip())


def passages():
    """Cards as search passages. Ids are negative so they never collide with Wikipedia passage ids."""
    out = []
    for i, c in enumerate(CARDS):
        out.append({"id": -(i + 1), "title": c["title"], "section": "Profile" if c["kind"] != "topic" else "Overview",
                    "text": text(c), "url": "", "card": True, "page": c["page"],
                    "aliases": ([c["n"].lower(), re.sub(r"^the ", "", c["n"].lower())] if c["n"] else []) + c["aliases"],
                    "keywords": c.get("keywords", "")})
    return out


def main():
    """Check every fact's keys against the card's own answer and against the scraped page."""
    chunks = json.loads((ROOT / "data" / "chunks.json").read_text())
    raw = {}
    for f in (ROOT / "data" / "raw").glob("*.json"):
        page = json.loads(f.read_text())["query"]["pages"][0]
        if page.get("extract"):
            raw[re.sub(r"\s+\((character|DC Comics|comics|comic book|identity|team)\)$", "", page["title"])] = page["extract"]
    titles = {c["title"] for c in chunks}
    n = bad = 0
    for c in CARDS:
        if c["page"] and c["page"] not in titles:
            print(f"NO PAGE  {c['title']!r}: page {c['page']!r} not in corpus")
        for field, a, k in facts(c):
            n += 1
            if not key_hit(a, k):
                bad += 1
                print(f"SELF     {c['title']} / {field}: keys {k!r} not in its own answer")
            if field in ("first", "by", "real") and c["page"] in raw and not key_hit(raw[c["page"]], k):
                bad += 1
                print(f"UNSOURCED {c['title']} / {field}: {k!r} not found on the page  ->  {a}")
    (ROOT / "data" / "cards.json").write_text(json.dumps(passages(), ensure_ascii=False, separators=(",", ":")))
    print(f"{len(CARDS)} cards, {n} facts, {bad} problems")


if __name__ == "__main__":
    main()
