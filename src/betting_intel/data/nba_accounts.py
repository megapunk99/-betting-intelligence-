"""
NBA Twitter/X accounts database — the definitive list of accounts to monitor
for actionable betting intelligence.

Every account is tagged with:
  - team(s) they cover
  - role (insider, beat_reporter, team_account, analyst, injury_tracker)
  - reliability score (0.0–1.0)
  - signal types they typically produce

These accounts are used by the TwitterSignalCollector to scrape and parse
real-time NBA intelligence that can give us an edge before sportsbooks
adjust their lines.

Signal types each account produces:
  - INJURY: "Player X is out tonight with ankle sprain"
  - LOAD_MGMT: "Player X sitting for rest on back-to-back"
  - LINEUP: "Coach says Player X will start at PF"
  - COACH_COMMENT: "Coach says Player X minutes will be limited"
  - BREAKING: "Player X traded to Team Y"
  - GTD: "Player X is a game-time decision"
  - PRACTICE: "Player X was a full participant in practice"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AccountRole(str, Enum):
    """Role of a Twitter/X account in the NBA reporting ecosystem."""

    INSIDER = "insider"              # National insiders (Shams, Woj, etc.)
    BEAT_REPORTER = "beat_reporter"   # Team-specific beat writers
    TEAM_ACCOUNT = "team_account"     # Official team Twitter accounts
    ANALYST = "analyst"              # Statistical / analytical accounts
    INJURY_TRACKER = "injury_tracker" # Injury-specific accounts
    ODDS_ACCOUNT = "odds_account"    # Sportsbook / odds accounts


class SignalType(str, Enum):
    """Types of actionable signals that can be extracted from tweets."""

    INJURY_OUT = "injury_out"                      # Player ruled OUT
    INJURY_QUESTIONABLE = "injury_questionable"    # Player questionable
    INJURY_PROBABLE = "injury_probable"            # Player probable
    INJURY_RETURN = "injury_return"               # Player returning from injury
    LOAD_MGMT = "load_management"                 # Player sitting for rest
    LINEUP_CHANGE = "lineup_change"               # Starting lineup change
    COACH_COMMENT = "coach_comment"               # Coach comment about usage
    GTD = "game_time_decision"                    # Game-time decision
    PRACTICE_STATUS = "practice_status"            # Practice participation
    TRADE = "trade"                                # Player traded
    CONTRACT = "contract"                          # Contract extension / signing
    SUSPENSION = "suspension"                     # Player suspended
    ILLNESS = "illness"                           # Player illness (non-injury)
    PERSONAL = "personal_reason"                  # Personal reasons absence
    WEATHER = "weather_delay"                     # Weather / travel issues
    LOCKER_ROOM = "locker_room"                   # Locker room atmosphere / morale
    CONDITIONING = "conditioning"                 # Conditioning / minutes limit


@dataclass
class NBAAccount:
    """A single NBA Twitter/X account to monitor for betting signals."""

    handle: str                          # Twitter handle (without @)
    name: str                            # Display name
    role: AccountRole                    # Role in the ecosystem
    teams: list[str] = field(default_factory=list)  # Teams covered (empty = league-wide)
    signal_types: list[SignalType] = field(default_factory=list)  # Signal types produced
    reliability: float = 0.7             # 0.0–1.0 reliability score
    notes: str = ""

    def __hash__(self):
        return hash(self.handle.lower())

    def __eq__(self, other):
        if isinstance(other, NBAAccount):
            return self.handle.lower() == other.handle.lower()
        return NotImplemented


# ═══════════════════════════════════════════════════════════════════════════
# NATIONAL INSIDERS — breaking news, trades, injuries
# ═══════════════════════════════════════════════════════════════════════════

NATIONAL_INSIDERS: list[NBAAccount] = [
    NBAAccount(
        handle="ShamsCharania",
        name="Shams Charania",
        role=AccountRole.INSIDER,
        teams=[],
        signal_types=[
            SignalType.INJURY_OUT, SignalType.INJURY_QUESTIONABLE,
            SignalType.INJURY_PROBABLE, SignalType.TRADE,
            SignalType.CONTRACT, SignalType.SUSPENSION,
            SignalType.LINEUP_CHANGE,
        ],
        reliability=0.95,
        notes="Most reliable NBA insider. First to break major news.",
    ),
    NBAAccount(
        handle="wojespn",
        name="Adrian Wojnarowski",
        role=AccountRole.INSIDER,
        teams=[],
        signal_types=[
            SignalType.INJURY_OUT, SignalType.INJURY_QUESTIONABLE,
            SignalType.INJURY_RETURN, SignalType.TRADE,
            SignalType.CONTRACT, SignalType.SUSPENSION,
        ],
        reliability=0.95,
        notes="Retired from reporting but archive is valuable. Replaced by Shams.",
    ),
    NBAAccount(
        handle="TheSteinLine",
        name="Marc Stein",
        role=AccountRole.INSIDER,
        teams=[],
        signal_types=[
            SignalType.INJURY_OUT, SignalType.COACH_COMMENT,
            SignalType.TRADE, SignalType.CONTRACT,
        ],
        reliability=0.85,
        notes="Veteran insider. Strong on coaching changes and team news.",
    ),
    NBAAccount(
        handle="ChrisBHaynes",
        name="Chris Haynes",
        role=AccountRole.INSIDER,
        teams=[],
        signal_types=[
            SignalType.INJURY_OUT, SignalType.INJURY_QUESTIONABLE,
            SignalType.COACH_COMMENT, SignalType.LOCKER_ROOM,
        ],
        reliability=0.80,
        notes="Strong player connections. Good for locker room insight.",
    ),
    NBAAccount(
        handle="WindhorstESPN",
        name="Brian Windhorst",
        role=AccountRole.INSIDER,
        teams=["Cavaliers", "Lakers", "Heat"],
        signal_types=[
            SignalType.COACH_COMMENT, SignalType.TRADE,
            SignalType.LOCKER_ROOM,
        ],
        reliability=0.78,
        notes="LeBron-connected insider. Good team culture intel.",
    ),
    NBAAccount(
        handle="BobbyMarks42",
        name="Bobby Marks",
        role=AccountRole.INSIDER,
        teams=[],
        signal_types=[
            SignalType.CONTRACT, SignalType.TRADE,
            SignalType.SUSPENSION,
        ],
        reliability=0.82,
        notes="Cap expert. Best for contract and trade machine analysis.",
    ),
    NBAAccount(
        handle="HDouglas83",
        name="Hoop Analysis Net",
        role=AccountRole.ANALYST,
        teams=[],
        signal_types=[
            SignalType.COACH_COMMENT, SignalType.LINEUP_CHANGE,
            SignalType.LOCKER_ROOM,
        ],
        reliability=0.70,
        notes="Advanced stats and lineup analysis.",
    ),
]

# ═══════════════════════════════════════════════════════════════════════════
# INJURY TRACKERS
# ═══════════════════════════════════════════════════════════════════════════

INJURY_TRACKERS: list[NBAAccount] = [
    NBAAccount(
        handle="UnderdogNBA",
        name="Underdog NBA",
        role=AccountRole.INJURY_TRACKER,
        teams=[],
        signal_types=[
            SignalType.INJURY_OUT, SignalType.INJURY_QUESTIONABLE,
            SignalType.INJURY_PROBABLE, SignalType.GTD,
            SignalType.PRACTICE_STATUS, SignalType.ILLNESS,
            SignalType.PERSONAL,
        ],
        reliability=0.85,
        notes="Excellent injury aggregation. Quick updates on player status.",
    ),
    NBAAccount(
        handle="FantasyLabsNBA",
        name="FantasyLabs NBA",
        role=AccountRole.INJURY_TRACKER,
        teams=[],
        signal_types=[
            SignalType.INJURY_OUT, SignalType.INJURY_QUESTIONABLE,
            SignalType.PRACTICE_STATUS, SignalType.GTD,
        ],
        reliability=0.82,
        notes="Fantasy-focused injury updates. Good for minute projections.",
    ),
    NBAAccount(
        handle="NBCSEdgeNBA",
        name="NBC Sports Edge NBA",
        role=AccountRole.INJURY_TRACKER,
        teams=[],
        signal_types=[
            SignalType.INJURY_OUT, SignalType.INJURY_QUESTIONABLE,
            SignalType.INJURY_PROBABLE, SignalType.COACH_COMMENT,
        ],
        reliability=0.75,
        notes="Injury aggregation + coach quotes. Solid secondary source.",
    ),
    NBAAccount(
        handle="Rotoworld_BBALL",
        name="Rotoworld Basketball",
        role=AccountRole.INJURY_TRACKER,
        teams=[],
        signal_types=[
            SignalType.INJURY_OUT, SignalType.INJURY_QUESTIONABLE,
            SignalType.LINEUP_CHANGE, SignalType.PRACTICE_STATUS,
        ],
        reliability=0.80,
        notes="Rotoworld player news. Veteran source with coach quotes.",
    ),
]

# ═══════════════════════════════════════════════════════════════════════════
# TEAM BEAT REPORTERS (most reliable for team-specific news)
# ═══════════════════════════════════════════════════════════════════════════

BEAT_REPORTERS: list[NBAAccount] = [
    # Atlantic Division
    NBAAccount("JaredWeissNBA", "Jared Weiss", AccountRole.BEAT_REPORTER,
               ["Celtics"], [SignalType.COACH_COMMENT, SignalType.LINEUP_CHANGE, SignalType.INJURY_OUT], 0.82),
    NBAAccount("SteveBHoop", "Steve Bulpett", AccountRole.BEAT_REPORTER,
               ["Celtics"], [SignalType.COACH_COMMENT, SignalType.LOCKER_ROOM], 0.78),
    NBAAccount("IanBegley", "Ian Begley", AccountRole.BEAT_REPORTER,
               ["Knicks"], [SignalType.INJURY_OUT, SignalType.COACH_COMMENT, SignalType.LINEUP_CHANGE], 0.83),
    NBAAccount("FredKatz", "Fred Katz", AccountRole.BEAT_REPORTER,
               ["Knicks"], [SignalType.COACH_COMMENT, SignalType.LINEUP_CHANGE, SignalType.LOCKER_ROOM], 0.80),
    NBAAccount("PompeyOnSixers", "Keith Pompey", AccountRole.BEAT_REPORTER,
               ["76ers"], [SignalType.INJURY_OUT, SignalType.COACH_COMMENT, SignalType.LINEUP_CHANGE], 0.78),
    NBAAccount("KyleNeubeck", "Kyle Neubeck", AccountRole.BEAT_REPORTER,
               ["76ers"], [SignalType.COACH_COMMENT, SignalType.TRADE, SignalType.LINEUP_CHANGE], 0.80),
    NBAAccount("blake_murphy", "Blake Murphy", AccountRole.BEAT_REPORTER,
               ["Raptors"], [SignalType.INJURY_OUT, SignalType.COACH_COMMENT, SignalType.LINEUP_CHANGE], 0.82),
    NBAAccount("MichaelGrange", "Michael Grange", AccountRole.BEAT_REPORTER,
               ["Raptors"], [SignalType.COACH_COMMENT, SignalType.TRADE, SignalType.LOCKER_ROOM], 0.78),
    NBAAccount("basketballtalk", "Kurt Helin", AccountRole.BEAT_REPORTER,
               ["Nets"], [SignalType.COACH_COMMENT, SignalType.INJURY_OUT], 0.72),

    # Central Division
    NBAAccount("KCJHoop", "K.C. Johnson", AccountRole.BEAT_REPORTER,
               ["Bulls"], [SignalType.INJURY_OUT, SignalType.COACH_COMMENT, SignalType.LINEUP_CHANGE], 0.85),
    NBAAccount("ChrisFedor", "Chris Fedor", AccountRole.BEAT_REPORTER,
               ["Cavaliers"], [SignalType.INJURY_OUT, SignalType.COACH_COMMENT, SignalType.LINEUP_CHANGE], 0.82),
    NBAAccount("JamesL EdwardsIII", "James L. Edwards III", AccountRole.BEAT_REPORTER,
               ["Pistons"], [SignalType.COACH_COMMENT, SignalType.LINEUP_CHANGE, SignalType.LOCKER_ROOM], 0.78),
    NBAAccount("ScottAgness", "Scott Agness", AccountRole.BEAT_REPORTER,
               ["Pacers"], [SignalType.INJURY_OUT, SignalType.COACH_COMMENT, SignalType.PRACTICE_STATUS], 0.83),
    NBAAccount("eric_nehm", "Eric Nehm", AccountRole.BEAT_REPORTER,
               ["Bucks"], [SignalType.INJURY_OUT, SignalType.COACH_COMMENT, SignalType.LINEUP_CHANGE], 0.80),

    # Southeast Division
    NBAAccount("ATLHawks beat", "Lauren L. Williams", AccountRole.BEAT_REPORTER,
               ["Hawks"], [SignalType.COACH_COMMENT, SignalType.LINEUP_CHANGE, SignalType.INJURY_OUT], 0.78),
    NBAAccount("RodBoone", "Rod Boone", AccountRole.BEAT_REPORTER,
               ["Hornets"], [SignalType.COACH_COMMENT, SignalType.INJURY_OUT, SignalType.LINEUP_CHANGE], 0.76),
    NBAAccount("BarryJackson", "Barry Jackson", AccountRole.BEAT_REPORTER,
               ["Heat"], [SignalType.INJURY_OUT, SignalType.COACH_COMMENT, SignalType.LINEUP_CHANGE], 0.82),
    NBAAccount("ByTimReynolds", "Tim Reynolds", AccountRole.BEAT_REPORTER,
               ["Heat"], [SignalType.COACH_COMMENT, SignalType.LOCKER_ROOM, SignalType.INJURY_OUT], 0.80),
    NBAAccount("JoshuaBRobbins", "Josh Robbins", AccountRole.BEAT_REPORTER,
               ["Magic"], [SignalType.INJURY_OUT, SignalType.COACH_COMMENT, SignalType.LINEUP_CHANGE], 0.80),
    NBAAccount("BulletsForever", "Sam Vecenie", AccountRole.BEAT_REPORTER,
               ["Wizards"], [SignalType.COACH_COMMENT, SignalType.LINEUP_CHANGE, SignalType.TRADE], 0.74),

    # Northwest Division
    NBAAccount("dnvr sports", "Mike Singer", AccountRole.BEAT_REPORTER,
               ["Nuggets"], [SignalType.INJURY_OUT, SignalType.COACH_COMMENT, SignalType.LINEUP_CHANGE], 0.80),
    NBAAccount("JonKrawczynski", "Jon Krawczynski", AccountRole.BEAT_REPORTER,
               ["Timberwolves"], [SignalType.INJURY_OUT, SignalType.COACH_COMMENT, SignalType.LINEUP_CHANGE], 0.82),
    NBAAccount("DarnellM", "Darnell Mayberry", AccountRole.BEAT_REPORTER,
               ["Thunder"], [SignalType.COACH_COMMENT, SignalType.LINEUP_CHANGE, SignalType.LOCKER_ROOM], 0.80),
    NBAAccount("Brandon Rahbar", "Brandon Rahbar", AccountRole.BEAT_REPORTER,
               ["Thunder"], [SignalType.INJURY_OUT, SignalType.COACH_COMMENT, SignalType.PRACTICE_STATUS], 0.78),
    NBAAccount("Joe Freeman", "Joe Freeman", AccountRole.BEAT_REPORTER,
               ["Trail Blazers"], [SignalType.INJURY_OUT, SignalType.COACH_COMMENT, SignalType.LINEUP_CHANGE], 0.80),
    NBAAccount("Tony Jones", "Tony Jones", AccountRole.BEAT_REPORTER,
               ["Jazz"], [SignalType.INJURY_OUT, SignalType.COACH_COMMENT, SignalType.LINEUP_CHANGE], 0.82),

    # Pacific Division
    NBAAccount("anthonyVslater", "Anthony Slater", AccountRole.BEAT_REPORTER,
               ["Warriors"], [SignalType.INJURY_OUT, SignalType.COACH_COMMENT, SignalType.LINEUP_CHANGE], 0.85),
    NBAAccount("Sam Amick", "Sam Amick", AccountRole.BEAT_REPORTER,
               ["Warriors", "Kings"], [SignalType.INJURY_OUT, SignalType.COACH_COMMENT, SignalType.LOCKER_ROOM], 0.82),
    NBAAccount("timbontemps", "Tim Bontemps", AccountRole.BEAT_REPORTER,
               ["Clippers"], [SignalType.COACH_COMMENT, SignalType.INJURY_OUT, SignalType.LINEUP_CHANGE], 0.80),
    NBAAccount("MikeTrudell", "Mike Trudell", AccountRole.BEAT_REPORTER,
               ["Lakers"], [SignalType.INJURY_OUT, SignalType.COACH_COMMENT, SignalType.LINEUP_CHANGE], 0.83),
    NBAAccount("Dan Woike", "Dan Woike", AccountRole.BEAT_REPORTER,
               ["Lakers"], [SignalType.COACH_COMMENT, SignalType.LOCKER_ROOM, SignalType.TRADE], 0.80),
    NBAAccount("DuaneRankin", "Duane Rankin", AccountRole.BEAT_REPORTER,
               ["Suns"], [SignalType.INJURY_OUT, SignalType.COACH_COMMENT, SignalType.LINEUP_CHANGE], 0.80),
    NBAAccount("Jason Jones", "Jason Jones", AccountRole.BEAT_REPORTER,
               ["Kings"], [SignalType.COACH_COMMENT, SignalType.LINEUP_CHANGE, SignalType.INJURY_OUT], 0.78),

    # Southwest Division
    NBAAccount("townbrad", "Brad Townsend", AccountRole.BEAT_REPORTER,
               ["Mavericks"], [SignalType.INJURY_OUT, SignalType.COACH_COMMENT, SignalType.LINEUP_CHANGE], 0.80),
    NBAAccount("Mark Berman", "Mark Berman", AccountRole.BEAT_REPORTER,
               ["Rockets"], [SignalType.INJURY_OUT, SignalType.COACH_COMMENT, SignalType.LINEUP_CHANGE], 0.82),
    NBAAccount("damichaelcole", "Damichael Cole", AccountRole.BEAT_REPORTER,
               ["Grizzlies"], [SignalType.INJURY_OUT, SignalType.COACH_COMMENT, SignalType.LINEUP_CHANGE], 0.78),
    NBAAccount("Will Guillory", "Will Guillory", AccountRole.BEAT_REPORTER,
               ["Pelicans"], [SignalType.INJURY_OUT, SignalType.COACH_COMMENT, SignalType.LINEUP_CHANGE], 0.82),
    NBAAccount("Tom Orsborn", "Tom Orsborn", AccountRole.BEAT_REPORTER,
               ["Spurs"], [SignalType.INJURY_OUT, SignalType.COACH_COMMENT, SignalType.LINEUP_CHANGE], 0.80),
]

# ═══════════════════════════════════════════════════════════════════════════
# OFFICIAL TEAM ACCOUNTS
# ═══════════════════════════════════════════════════════════════════════════

TEAM_ACCOUNTS: list[NBAAccount] = [
    NBAAccount("celtics", "Boston Celtics", AccountRole.TEAM_ACCOUNT,
               ["Celtics"], [SignalType.INJURY_OUT, SignalType.PRACTICE_STATUS], 0.90),
    NBAAccount("nyknicks", "New York Knicks", AccountRole.TEAM_ACCOUNT,
               ["Knicks"], [SignalType.INJURY_OUT, SignalType.PRACTICE_STATUS], 0.90),
    NBAAccount("sixers", "Philadelphia 76ers", AccountRole.TEAM_ACCOUNT,
               ["76ers"], [SignalType.INJURY_OUT, SignalType.PRACTICE_STATUS], 0.90),
    NBAAccount("Raptors", "Toronto Raptors", AccountRole.TEAM_ACCOUNT,
               ["Raptors"], [SignalType.INJURY_OUT, SignalType.PRACTICE_STATUS], 0.90),
    NBAAccount("BrooklynNets", "Brooklyn Nets", AccountRole.TEAM_ACCOUNT,
               ["Nets"], [SignalType.INJURY_OUT, SignalType.PRACTICE_STATUS], 0.90),
    NBAAccount("chicagobulls", "Chicago Bulls", AccountRole.TEAM_ACCOUNT,
               ["Bulls"], [SignalType.INJURY_OUT, SignalType.PRACTICE_STATUS], 0.90),
    NBAAccount("cavs", "Cleveland Cavaliers", AccountRole.TEAM_ACCOUNT,
               ["Cavaliers"], [SignalType.INJURY_OUT, SignalType.PRACTICE_STATUS], 0.90),
    NBAAccount("DetroitPistons", "Detroit Pistons", AccountRole.TEAM_ACCOUNT,
               ["Pistons"], [SignalType.INJURY_OUT, SignalType.PRACTICE_STATUS], 0.90),
    NBAAccount("Pacers", "Indiana Pacers", AccountRole.TEAM_ACCOUNT,
               ["Pacers"], [SignalType.INJURY_OUT, SignalType.PRACTICE_STATUS], 0.90),
    NBAAccount("Bucks", "Milwaukee Bucks", AccountRole.TEAM_ACCOUNT,
               ["Bucks"], [SignalType.INJURY_OUT, SignalType.PRACTICE_STATUS], 0.90),
    NBAAccount("ATLHawks", "Atlanta Hawks", AccountRole.TEAM_ACCOUNT,
               ["Hawks"], [SignalType.INJURY_OUT, SignalType.PRACTICE_STATUS], 0.90),
    NBAAccount("hornets", "Charlotte Hornets", AccountRole.TEAM_ACCOUNT,
               ["Hornets"], [SignalType.INJURY_OUT, SignalType.PRACTICE_STATUS], 0.90),
    NBAAccount("MiamiHEAT", "Miami Heat", AccountRole.TEAM_ACCOUNT,
               ["Heat"], [SignalType.INJURY_OUT, SignalType.PRACTICE_STATUS], 0.90),
    NBAAccount("OrlandoMagic", "Orlando Magic", AccountRole.TEAM_ACCOUNT,
               ["Magic"], [SignalType.INJURY_OUT, SignalType.PRACTICE_STATUS], 0.90),
    NBAAccount("WashWizards", "Washington Wizards", AccountRole.TEAM_ACCOUNT,
               ["Wizards"], [SignalType.INJURY_OUT, SignalType.PRACTICE_STATUS], 0.90),
    NBAAccount("nuggets", "Denver Nuggets", AccountRole.TEAM_ACCOUNT,
               ["Nuggets"], [SignalType.INJURY_OUT, SignalType.PRACTICE_STATUS], 0.90),
    NBAAccount("Timberwolves", "Minnesota Timberwolves", AccountRole.TEAM_ACCOUNT,
               ["Timberwolves"], [SignalType.INJURY_OUT, SignalType.PRACTICE_STATUS], 0.90),
    NBAAccount("okcthunder", "OKC Thunder", AccountRole.TEAM_ACCOUNT,
               ["Thunder"], [SignalType.INJURY_OUT, SignalType.PRACTICE_STATUS], 0.90),
    NBAAccount("trailblazers", "Portland Trail Blazers", AccountRole.TEAM_ACCOUNT,
               ["Trail Blazers"], [SignalType.INJURY_OUT, SignalType.PRACTICE_STATUS], 0.90),
    NBAAccount("utahjazz", "Utah Jazz", AccountRole.TEAM_ACCOUNT,
               ["Jazz"], [SignalType.INJURY_OUT, SignalType.PRACTICE_STATUS], 0.90),
    NBAAccount("warriors", "Golden State Warriors", AccountRole.TEAM_ACCOUNT,
               ["Warriors"], [SignalType.INJURY_OUT, SignalType.PRACTICE_STATUS], 0.90),
    NBAAccount("LAClippers", "LA Clippers", AccountRole.TEAM_ACCOUNT,
               ["Clippers"], [SignalType.INJURY_OUT, SignalType.PRACTICE_STATUS], 0.90),
    NBAAccount("Lakers", "Los Angeles Lakers", AccountRole.TEAM_ACCOUNT,
               ["Lakers"], [SignalType.INJURY_OUT, SignalType.PRACTICE_STATUS], 0.90),
    NBAAccount("Suns", "Phoenix Suns", AccountRole.TEAM_ACCOUNT,
               ["Suns"], [SignalType.INJURY_OUT, SignalType.PRACTICE_STATUS], 0.90),
    NBAAccount("SacramentoKings", "Sacramento Kings", AccountRole.TEAM_ACCOUNT,
               ["Kings"], [SignalType.INJURY_OUT, SignalType.PRACTICE_STATUS], 0.90),
    NBAAccount("dallasmavs", "Dallas Mavericks", AccountRole.TEAM_ACCOUNT,
               ["Mavericks"], [SignalType.INJURY_OUT, SignalType.PRACTICE_STATUS], 0.90),
    NBAAccount("HoustonRockets", "Houston Rockets", AccountRole.TEAM_ACCOUNT,
               ["Rockets"], [SignalType.INJURY_OUT, SignalType.PRACTICE_STATUS], 0.90),
    NBAAccount("memgrizz", "Memphis Grizzlies", AccountRole.TEAM_ACCOUNT,
               ["Grizzlies"], [SignalType.INJURY_OUT, SignalType.PRACTICE_STATUS], 0.90),
    NBAAccount("PelicansNBA", "New Orleans Pelicans", AccountRole.TEAM_ACCOUNT,
               ["Pelicans"], [SignalType.INJURY_OUT, SignalType.PRACTICE_STATUS], 0.90),
    NBAAccount("spurs", "San Antonio Spurs", AccountRole.TEAM_ACCOUNT,
               ["Spurs"], [SignalType.INJURY_OUT, SignalType.PRACTICE_STATUS], 0.90),
]

# ═══════════════════════════════════════════════════════════════════════════
# ODDS & ANALYTICS ACCOUNTS
# ═══════════════════════════════════════════════════════════════════════════

ODDS_ACCOUNTS: list[NBAAccount] = [
    NBAAccount("ActionNetworkHQ", "The Action Network", AccountRole.ODDS_ACCOUNT,
               [], [SignalType.INJURY_OUT, SignalType.COACH_COMMENT, SignalType.LINEUP_CHANGE], 0.80,
               "Aggregates betting news and line movements."),
    NBAAccount("SportsbookScout", "Sportsbook Scout", AccountRole.ODDS_ACCOUNT,
               [], [SignalType.INJURY_OUT, SignalType.COACH_COMMENT], 0.75,
               "Line movement and sharp money tracking."),
    NBAAccount("EVanalytics", "EV Analytics", AccountRole.ODDS_ACCOUNT,
               [], [SignalType.INJURY_OUT, SignalType.COACH_COMMENT], 0.72,
               "EV-based betting insights."),
]

# ═══════════════════════════════════════════════════════════════════════════
# AGGREGATED ACCOUNT DATABASE
# ═══════════════════════════════════════════════════════════════════════════

def get_all_accounts() -> list[NBAAccount]:
    """Return ALL tracked NBA accounts."""
    return (
        NATIONAL_INSIDERS + INJURY_TRACKERS + BEAT_REPORTERS
        + TEAM_ACCOUNTS + ODDS_ACCOUNTS
    )


def get_accounts_by_team(team_name: str) -> list[NBAAccount]:
    """Get all accounts that cover a specific team."""
    team_lower = team_name.lower()
    return [
        acc for acc in get_all_accounts()
        if any(t.lower() == team_lower for t in acc.teams)
    ]


def get_accounts_by_role(role: AccountRole) -> list[NBAAccount]:
    """Get all accounts with a specific role."""
    return [acc for acc in get_all_accounts() if acc.role == role]


def get_accounts_by_signal(signal: SignalType) -> list[NBAAccount]:
    """Get all accounts that produce a specific signal type."""
    return [acc for acc in get_all_accounts() if signal in acc.signal_types]


def get_high_reliability_accounts(min_reliability: float = 0.8) -> list[NBAAccount]:
    """Get accounts above a reliability threshold."""
    return [acc for acc in get_all_accounts() if acc.reliability >= min_reliability]


# ── Display helpers ───────────────────────────────────────────────────────

def format_account(account: NBAAccount) -> str:
    """Format an account for display."""
    teams = ", ".join(account.teams) if account.teams else "League-wide"
    signals = ", ".join(s.value for s in account.signal_types[:4])
    return (
        f"@{account.handle:<22} | {account.role.value:<16} | "
        f"Rel: {account.reliability:.0%} | Teams: {teams:<20} | {signals}"
    )


def print_accounts(accounts: Optional[list[NBAAccount]] = None) -> None:
    """Pretty-print a list of accounts."""
    if accounts is None:
        accounts = get_all_accounts()
    for acc in sorted(accounts, key=lambda a: (a.role.value, a.handle)):
        print(format_account(acc))
    print(f"\nTotal: {len(accounts)} accounts")
