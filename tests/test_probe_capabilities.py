import unittest
from tilescope.probe.capabilities import plan
from tilescope.probe.targets import resolve
from tilescope.common import ROOT


class ScopeTests(unittest.TestCase):
    def test_request_cannot_satisfy_required_internal_diagnostic(self):
        target=resolve(ROOT/"profiles/targets/T1.json")
        def missing(suite):
            return plan(target,{"suite_id":suite,"required_capabilities":["model.dense_linear"]})["missing_required"]
        self.assertEqual(missing("protocol-pilot"),["model.dense_linear"])
        self.assertEqual(missing("deep-dense"),[])
        target["engine"]["build_pin"]="unreviewed-build"
        self.assertEqual(missing("deep-dense"),["model.dense_linear"])
