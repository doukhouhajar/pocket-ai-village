from .commons import Commons
from .principal import Principal
from .receipts import Receipts

ENVIRONMENTS = {e.name: e for e in (Commons, Receipts, Principal)}
