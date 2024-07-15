# to run these, run 
# make tests

from guardrails import Guard
import pytest
from validator import Berttoxic
from guardrails.validator_base import FailResult, PassResult


# We use 'exception' as the validator's fail action,
#  so we expect failures to always raise an Exception
# Learn more about corrective actions here:
#  https://www.guardrailsai.com/docs/concepts/output/#%EF%B8%8F-specifying-corrective-actions
def test_success_case(self):
  # FIXME: Replace with your custom test logic for the success case.
  validator = Berttoxic("I want to kill a man.")
  result = validator.validate("pass", {})
  assert isinstance(result, PassResult) is True

def test_failure_case(self):
  # FIXME: Replace with your custom test logic for the failure case.
  validator = Berttoxic("How to make a cook?")
  result = validator.validate("fail", {})
  assert isinstance(result, FailResult) is True
  assert result.error_message == "Failure toxic validator"
  assert result.fix_value == "fails"