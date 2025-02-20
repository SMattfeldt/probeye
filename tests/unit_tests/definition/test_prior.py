# standard library imports
import unittest

# local imports
from probeye.definition.prior import PriorBase


class TestProblem(unittest.TestCase):
    def test_prior_template(self):
        prior_template = PriorBase(
            "a", ["loc_a", "scale_a"], "a_normal", "normal distribution"
        )
        # check that the attributes are wired correctly
        self.assertEqual(prior_template.ref_prm, "a")
        self.assertEqual(prior_template.name, "a_normal")
        self.assertEqual(prior_template.prior_type, "normal distribution")
        self.assertEqual(
            prior_template.prms_def, {"a": "a", "loc_a": "loc_a", "scale_a": "scale_a"}
        )
        self.assertEqual(
            prior_template.hyperparameters, {"loc_a": "loc_a", "scale_a": "scale_a"}
        )
        # check that the __str__-method works
        self.assertTrue(len(prior_template.__str__()) > 0)


if __name__ == "__main__":
    unittest.main()
