"""Small first-name -> gender lexicon (common English-language fiction names). Used only as one vote among several."""

from typing import Optional

FEMALE_NAMES = set("""

abby abigail ada adele adriana agnes alexandra alice alicia alina alison allison alyssa amanda amber amelia amy
ana anastasia andrea angela angelica anna annabel anne annie april ariel ashley audrey aurora ava barbara beatrice
becca becky bella beth bethany betty beverly bianca bonnie brenda bridget brittany brooke caitlin camila carla carmen
carol caroline carolyn cassandra catherine cecilia celeste charlotte chelsea cheryl chloe christina christine cindy
claire clara clare claudia colleen cora courtney crystal cynthia daisy dana daphne darcy dawn deborah delilah denise
diana diane donna dora dorothy edith eileen elaine eleanor elena eliza elizabeth ella ellen ellie eloise elsa emily
emma erica erin esther eva eve evelyn faith fiona florence frances freya gabriella gabrielle gemma georgia gina gloria
grace gwen hailey hannah harper harriet hazel heather heidi helen holly hope ingrid irene iris isabel isabella isla ivy
jackie jacqueline jade jane janet janice jasmine jean jennifer jenny jessica jill joan joanna jocelyn jodie josephine
joy joyce judith julia juliana julie june kara karen kate katherine kathleen kathryn katie kayla kelly kim kimberly
kira kristen kristina laura lauren layla leah lena leona leslie lila lillian lily linda lisa lois lola lorraine lucia
lucy luna lydia lynn mabel madeline madison maggie mara margaret maria marie marian marilyn marina martha mary maya
megan melanie melissa mia michelle mildred mila miranda miriam molly monica nadia naomi natalie natasha nell nellie
nicole nina nora olive olivia paige pamela patricia paula pauline penelope phoebe phyllis piper priscilla rachel
rebecca regina renee rita roberta robin rosa rose rosemary ruby ruth sabrina sadie sally samantha sandra sara sarah
savannah scarlett selena serena shannon sharon sheila shelby shirley sienna sofia sophia sophie stella stephanie
susan suzanne sylvia tabitha tamara tanya tara teresa tess tessa theresa tiffany tina tracy trinity ursula valerie
vanessa vera veronica verity victoria violet virginia vivian wanda wendy whitney willow winifred yvonne zoe zoey
""".split())

MALE_NAMES = set("""

aaron abel abraham adam adrian alan albert alec alex alexander alfred alistair allen andre andrew andy angus anthony
archie arnold arthur austin barry ben benjamin bernard bill billy blake bob bobby brad bradley brandon brian bruce bryan
caleb calvin cameron carl carlos casey cedric charles charlie chase chris christian christopher clarence clark claude
clayton cliff clifford clint cody colin connor conrad corey craig curtis dale damian damon dan daniel danny darren dave
david dean dennis derek devin dexter dominic donald douglas drew duncan dustin dylan earl ed eddie edgar edmund edward
edwin elijah elliot elliott ellis emmett eric ernest ethan eugene evan felix finn francis frank franklin fred frederick
gabriel gary gavin gene geoffrey george gerald gideon gilbert glenn gordon graham grant greg gregory griffin gus guy
hal hank harold harry harvey heath henry herbert howard hugh hugo ian isaac isaiah ivan jack jackson jacob jake james
jamie jared jason jasper jay jeff jeffrey jeremy jerome jerry jesse jim jimmy joe joel john johnny jonah jonathan
jordan joseph josh joshua julian justin keith kelvin ken kenneth kevin kyle lance larry lawrence lee leo leon leonard
leroy lewis liam lincoln logan louis lucas luke malcolm marcus mark martin marshall matt matthew maurice max maxwell
michael mike miles mitchell morgan nathan nathaniel neil nicholas nick noah noel norman oliver oscar otis owen patrick
paul peter philip phil preston quentin ralph randall randy ray raymond reed rex richard rick riley rob robert robin
rodney roger roland ronald ross roy russell ryan sam samuel scott sean sebastian seth shane shaun simon spencer stanley
stephen steve steven stuart ted terry theo theodore thomas tim timothy toby todd tom tommy tony travis trevor tristan
troy tyler vernon victor vincent walter warren wayne wesley wilfred will william willie wyatt xavier zachary zack
""".split())

# A handful of common fiction names absent from the block lists above.
_EXTRA_FEMALE = """
marjorie margery rachael winnie hattie gwendolen gwendolyn eunice adeline coraline
prudence dorcas tabitha lottie mattie nell pollyanna honoria lavinia letitia lucinda
mabel maud maude minnie nora phoebe rosalind rowena sylvia theodora ursula veronica
wilhelmina gertrude agatha beatrix clementine constance cordelia ethel eudora flora
imogen isadora josephine juliet kitty lydia marianne mathilda mildred miranda ophelia
penelope priscilla rhoda sabrina selina tess theresa viola wanda
""".split()

_EXTRA_MALE = """
cecil randolph ronny eugenio basil dickon silas ebenezer abner algernon ambrose
augustus barnaby bartholomew clive cuthbert cyrus desmond edgar edmund egbert elias
ephraim eustace ezra ferdinand godfrey harold hector horace hubert humphrey isaac
ivan jeremiah josiah lancelot leopold lionel lucius marmaduke milo mortimer neville
nigel octavius oswald percival reginald roderick rupert tobias titus wallace wilbur
winston
""".split()

FEMALE_NAMES |= set(_EXTRA_FEMALE)
MALE_NAMES |= set(_EXTRA_MALE)


def name_gender(token: str) -> Optional[str]:
    """Return 'male' / 'female' / None for a single first-name token."""
    t = token.strip(".,\"'").lower()
    if t in FEMALE_NAMES and t not in MALE_NAMES:
        return "female"
    if t in MALE_NAMES and t not in FEMALE_NAMES:
        return "male"
    return None
