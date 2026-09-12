"""Model-as-judge scoring for properties only a reader can assess.

For the properties that are real but not mechanical: whether a draft
sounds like the person it was written for, whether a response actually
answered the question, whether it drifted into generic register.

Responsible for:
    - Asking a model whether a stated property holds for a response.
    - Returning the verdict together with the model's stated reason.

Deliberately not responsible for:
    - Being treated as ground truth. The judge is another model with its
      own failure modes, and its agreement with a human label is itself
      worth measuring.
    - Open-ended quality opinions. A judged property is specific and
      written down in the case, not "is this good".
    - Hiding its cost. Judge runs cost money and time, and the report
      surfaces both.
"""
