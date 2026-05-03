"""Bidder model implementations."""

from .linear_bidder import LinearBidder
from .ta_pid import TAPIDBidder
from .m_pid import MPIDBidder
from .mystique import Mystique
from .broi_bidder import BROI
from .rlb_dp_bidder import RLBDPBidder

try:
    from .drlb_bidder import DRLBBidder
except ModuleNotFoundError:
    DRLBBidder = None

BIDDERS: dict[str, type] = {
    "linear": LinearBidder,
    "ta_pid": TAPIDBidder,
    "m_pid": MPIDBidder,
    "mystique": Mystique,
    "broi": BROI,
    "rlb_dp": RLBDPBidder,
}

if DRLBBidder is not None:
    BIDDERS["drlb"] = DRLBBidder
