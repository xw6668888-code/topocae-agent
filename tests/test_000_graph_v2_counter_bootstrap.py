"""Discovery-order compatibility for the Graph v2 representation tests.

The Windows workspace sandbox currently prevents in-place edits of an existing
file.  Import the representation test module first and provide its missing
``Counter`` global so both targeted and full discovery runs exercise the same
test body.  Remove this shim after the source file can be edited in place.
"""

from collections import Counter

import test_graph_v2_representation as graph_v2_representation_tests


graph_v2_representation_tests.Counter = Counter

