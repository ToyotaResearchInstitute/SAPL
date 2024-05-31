# Copyright 2023 Toyota Research Institute.  All rights reserved.
"""
This modules constructs a computation graph for WSTL weighted robustness
and performs the computation. It returns the weighted robustness values
of a set of signals with different weight valuations. These weight
valuations can be set as all 1 (STL case) or can be set randomly.

Implementation is inspired from STLCG code base.
See: https://github.com/StanfordASL/stlcg

Author: Ruya Karagulle
Date: May 2023
"""

import torch
import numpy as np

LARGE_NUMBER = 10**6


class Maxish(torch.nn.Module):
    """Defines max and softmax function."""

    def __init__(self):
        super(Maxish, self).__init__()

    def forward(self, x, scale: float, axis: int):
        """
        Forward pass of the class.

        Args:
            x (torch.Tensor or Expression): signal to take the max over
            scale: scaling factor for softmax. If scale = -1 it is the max function.
            axis (int): axis to take the max on

        Return:
            max_value (torch.Tensor): max value. Dimensions kept same
        """

        if isinstance(x, Expression):
            assert x.value is not None, "Input Expression does not have numerical values."
            x = x.value
        if scale > 0:
            max_value = torch.logsumexp(x * scale, dim=axis, keepdim=True) / scale
        else:
            max_value = torch.max(x, axis=axis, keepdim=True)[0]
        return max_value


class Minish(torch.nn.Module):
    """Defines min and softmin function."""

    def __init__(self, name="Minish input"):
        super(Minish, self).__init__()
        self.input_name = name

    def forward(self, x, scale, axis):
        """
        Forward pass of the class.

        Args:
            x (torch.Tensor or Expression): signal to take the min over
            scale: scaling factor for softmax. If scale = -1 it is the min function.
            axis (int): axis to take the min on

        Return:
            min_value (torch.Tensor): min value. Dimensions kept same
        """

        if isinstance(x, Expression):
            assert x.value is not None, "Input Expression does not have numerical values."
            x = x.value
        if scale > 0:
            min_value = -torch.logsumexp(-x * scale, dim=axis, keepdim=True) / scale
        else:
            min_value = torch.min(x, axis=axis, keepdim=True)[0]
        return min_value


class WSTL_Formula(torch.nn.Module):
    """Defines an WSTL formula.

    Attributes:
        weights (torch.nn.ParameterDict): weight valuations associated
                                          with each subformula

    Methods:
        robustness: Computes robustness for a given time instance.
        set_weights: Initializes weight values.
        update_weights: Updates weight values for sublayers.
        forward: Computes weighted robustness for given input signals
                 and for all weight valuations.
    """

    def __init__(self):
        super(WSTL_Formula, self).__init__()
        self.weights = torch.nn.ParameterDict({})

    def robustness(self, inputs, scale=-1, t=0):
        """
        Returns WSTL weighted robustness value for given input signals
        and for all weight valuation samples at t=0, by default.
        Note that robustness is computed per each time instant.

        Args:
            inputs (Expression or torch.Tensor or tuple): Input signals for robustness.
            scale (int): Scaling factor for robustness computation.
            t (int): Time instance for which to compute the robustness.

        Returns:
            torch.Tensor: WSTL weighted robustness values.
        """
        return self.forward(inputs, scale=scale)[:, t, :, :].unsqueeze(1)

    def set_weights(self, inputs, w_range, no_samples, random=False, seed=None, **kwargs):
        """
        Initializes weight values.
        If random = False, it initializes weights at 1.
        If random = True, it initializes weights uniformly random between given range.

        Args:
            inputs (Expression or torch.Tensor or tuple): Input signals.
            w_range (tuple): Weight range for random initialization.
            no_samples (int): Number of weight valuation samples to be set
                              (useful for random case).
            random (bool): Flag for random initialization.
            seed (int): Seed for reproducibility.
            **kwargs: Additional keyword arguments.
        """
        if seed is not None:
            torch.manual_seed(seed)
        self.weight_assignment(inputs, w_range, no_samples, random, **kwargs)

    def update_weights(self):
        """
        Updates weight values for sublayers.
        When a change in weight values of the main formula is on effect,
        this change needs be executed for sublayers.
        This function takes the main formula, and applies the changes on sublayers.
        """
        self.weight_update()

    def forward(formula, inputs, **kwargs):
        """
        Computes weighted robustness for input signals and for all weight valuations.

        Args:
            inputs (Expression or torch.Tensor or tuple): Input signals.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Weighted robustness values.
        """
        try:
            sc = kwargs["scale"]
        except KeyError:
            sc = -1

        # checks if inputs are in correct form.
        if isinstance(inputs, Expression):
            if inputs.value is None:
                raise TypeError("Input Expression does not have numerical values")
            return formula.robustness_value(inputs.value, scale=sc)
        elif isinstance(inputs, torch.Tensor):
            return formula.robustness_value(inputs, scale=sc)
        elif isinstance(inputs, tuple):
            return formula.robustness_value(convert_inputs(inputs), scale=sc)
        else:
            raise ValueError("Invalid input trace")


class Temporal_Operator(WSTL_Formula):
    """
    Defines Temporal operators in the syntax: Always, Eventually.
    Until is defined separately.

    Attributes:
        interval (Interval): Time interval associated with the temporal operator.
        subformula (WSTL_Formula): Subformula encapsulated by the temporal operator.
        operation (None): Placeholder for the specific temporal operation.

    Methods:
        weight_assignment: Assigns weight values to temporal operators given a range.
        weight_update: Updates weights in temporal operators.
        robustness_value: Computes weighted robustness of temporal operators.
    """

    def __init__(self, subformula, interval=None):
        """
        Initialize a Temporal_Operator.

        Args:
            subformula (WSTL_Formula): Subformula encapsulated by the temporal operator.
            interval (list or None): Time interval associated with the temporal operator.
                                     None for operators with [0, infinity] interval.
        """

        if not isinstance(subformula, WSTL_Formula):
            raise TypeError("Subformula needs to be an STL formula.")
        if interval is not None and len(interval) != 2:
            raise TypeError("Interval should be None or a list of length 2.")

        super(Temporal_Operator, self).__init__()
        self.interval = Interval(interval)
        self.subformula = subformula
        self.operation = None

    def weight_assignment(self, inputs, w_range, no_samples, random, **kwargs):
        """
        Assigns weight values to temporal operators given a range.

        Args:
            inputs (Expression or torch.Tensor): Input signals.
            w_range (tuple): Weight range for random initialization.
            no_samples (int): Number of weight valuation samples to be set
                              (useful for random case).
            random (bool): Flag for random initialization.
            **kwargs: Additional keyword arguments.
        """

        try:
            sc = kwargs["scale"]
        except KeyError:
            sc = -1
        self.subformula.weight_assignment(inputs, w_range, no_samples, random, **kwargs)
        trace = self.subformula(inputs, scale=sc)
        self.interval.set_interval(trace.shape[1])

        for keys in self.subformula.weights.keys():
            self.weights[keys] = self.subformula.weights[keys]
        self.compute_weights(w_range, no_samples, random)

    def weight_update(self):
        """Updates weights in temporal operators."""

        for keys in self.subformula.weights.keys():
            self.subformula.weights[keys] = self.weights[keys]
        self.subformula.weight_update()

    def robustness_value(self, inputs, **kwargs):
        """
        Computes weighted robustness of temporal operators.

        Args:
            inputs (Expression or torch.Tensor or tuple): Input signals.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Weighted robustness values
        """

        try:
            sc = kwargs["scale"]
        except KeyError:
            sc = -1
        trace = self.subformula(inputs, scale=sc)
        outputs = self.compute_robustness(trace, scale=sc)
        return outputs


class Logic_Operator(WSTL_Formula):
    """
    Defines Logic Operators in the syntax: And, Or.

    Attributes:
        subformula1 (WSTL_Formula): First subformula.
        subformula2 (WSTL_Formula): Second subformula.
        operation (None): Placeholder for the specific logic operation.

    Methods:
        weight_assignment: Assigns weight values to logic operators given a range.
        weight_update: Updates weights in logic operators.
        robustness_value: Computes weighted robustness of logic operators.
    """

    def __init__(self, subformula1, subformula2):
        """
        Initialize a Logic_Operator.

        Args:
            subformula1 (WSTL_Formula): First subformula.
            subformula2 (WSTL_Formula): Second subformula.
        """
        if not isinstance(subformula1, WSTL_Formula):
            raise TypeError("Subformula1 needs to be an STL formula.")
        if not isinstance(subformula2, WSTL_Formula):
            raise TypeError("Subformula2 needs to be an STL formula.")

        super(Logic_Operator, self).__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2
        self.operation = None

    def weight_assignment(self, inputs, w_range, no_samples, random, **kwargs):
        """
        Assigns weight values to logic operators given a range.

        Args:
            inputs (tuple): Tuple containing input signals for subformula 1 and 2.
            w_range (tuple): Weight range for random initialization.
            no_samples (int): Number of weight valuation samples to be set
                              (useful for random case).
            random (bool): Flag for random initialization.
            **kwargs: Additional keyword arguments.
        """

        self.subformula1.weight_assignment(
            inputs[0], w_range, no_samples, random, **kwargs
        )
        self.subformula2.weight_assignment(
            inputs[1], w_range, no_samples, random, **kwargs
        )
        for keys in self.subformula1.weights.keys():
            self.weights[keys] = self.subformula1.weights[keys]
        for keys in self.subformula2.weights.keys():
            self.weights[keys] = self.subformula2.weights[keys]
        self.compute_weights(w_range, no_samples, random)

    def weight_update(self):
        """Updates weights in logic operators."""

        for keys in self.subformula1.weights.keys():
            self.subformula1.weights[keys] = self.weights[keys]
        for keys in self.subformula2.weights.keys():
            self.subformula2.weights[keys] = self.weights[keys]
        self.subformula1.weight_update()
        self.subformula2.weight_update()

    def robustness_value(self, inputs, **kwargs):
        """
        Computes weighted robustness of logic operators.

        Args:
            inputs (tuple): Tuple containing input signals for subformula 1 and 2.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Weighted robustness values.
        """

        try:
            sc = kwargs["scale"]
        except KeyError:
            sc = -1

        trace1 = self.subformula1(inputs[0], scale=sc)
        trace2 = self.subformula2(inputs[1], scale=sc)
        trace_length = min(trace1.shape[1], trace2.shape[1])

        trace = torch.cat(
            [trace1[:, :trace_length, :], trace2[:, :trace_length, :]], axis=-1
        )

        outputs = self.compute_robustness(trace, scale=sc)
        return outputs


class Predicate(WSTL_Formula):
    """Defines predicates."""

    def __init__(self, lhs, val=0):
        """
        Initialize a Predicate.

        Args:
            lhs (Expression): Left-hand side expression of the predicate.
            val: Right-hand side value of the predicate.
                Should not be a string.
        """
        if isinstance(val, str):
            raise TypeError("RHS value cannot be a string.")
        if not isinstance(lhs, Expression):
            raise TypeError("LHS should be an Expression.")

        super(Predicate, self).__init__()
        self.lhs = lhs
        self.val = val

    def weight_assignment(self, input, w_range, no_samples, random, **kwargs):
        """
        Assigns weight values to the predicate given a range.

        As there is no weight in predicates, this function returns nothing.
        """

        pass

    def weight_update(self):
        """
        Updates weights in the predicate.

        As there is no weight in predicates, this function returns nothing.
        """

        pass

    def robustness_value(self, inputs, **kwargs):
        """
        Computes weighted robustness of the predicate.

        Args:
            inputs (Expression or torch.Tensor): Input signals.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Weighted robustness values.
        """

        scale = kwargs["scale"]
        return self.calculate_robustness(inputs, scale=scale)


class BoolTrue(Predicate):
    """
    Defines Boolean True. Note that Boolean True is not a predicate.
    Comparison value is left undefined.

    Attributes:
        lhs: Left-hand side expression.

    Methods:
        robustness_value: Computes the weighted robustness of the Boolean True.
        __str__: Returns a string representation of the BoolTrue.
    """

    def __init__(self, lhs):
        """
        Initialize a BoolTrue.

        Args:
            lhs: Left-hand side expression.
        """

        super(BoolTrue, self).__init__(lhs)

    def robustness_value(self, trace, **kwargs):
        """
        Computes the weighted robustness of the Boolean True.

        Args:
            trace (Expression or torch.Tensor): Input signal trace.

        Returns:
            torch.Tensor: Weighted robustness values.
        """
        if isinstance(trace, Expression):
            trace = trace.value
        return LARGE_NUMBER * trace

    def __str__(self):
        """Returns a string representation of the BoolTrue."""

        lhs_str = self.lhs
        if isinstance(self.lhs, Expression):
            lhs_str = self.lhs.name
        return lhs_str


class LessThan(Predicate):
    """
    Defines the predicate with less than or equal to comparison:
    "lhs <= val" where lhs is the signal, and val is the constant.

    Attributes:
        lhs: Left-hand side expression or signal.
        val: Right-hand side constant value for the comparison.

    Methods:
        robustness_value: Computes the weighted robustness of the "less than" predicate.
        __str__: Returns a string representation of the LessThan instance.
    """

    def __init__(self, lhs, val):
        """
        Initialize a LessThan instance.

        Args:
            lhs (str or Expression): Left-hand side expression or signal.
            val (float, int, Expression, torch.Tensor): Right-hand side constant value
                                                        for the comparison.
        """

        super(LessThan, self).__init__(lhs, val)

    def robustness_value(self, trace, **kwargs):
        """
        Computes the weighted robustness of the "less than" predicate.

        Args:
            trace (Expression or torch.Tensor): Input signal trace.

        Returns:
            torch.Tensor: Weighted robustness values.
        """

        if isinstance(trace, Expression):
            trace = trace.value
        if isinstance(self.val, Expression):
            return self.val.value - trace
        else:
            return self.val - trace

    def __str__(self):
        """Returns a string representation of the LessThan instance."""
        lhs_str = self.lhs
        if isinstance(self.lhs, Expression):
            lhs_str = self.lhs.name
        if isinstance(self.val, str):
            return lhs_str + " <= " + self.val
        if isinstance(self.val, Expression):
            return lhs_str + " <= " + self.val.name
        if isinstance(self.val, torch.Tensor):
            return lhs_str + " <= " + tensor2str(self.val)
        return lhs_str + " <= " + str(self.val)


class GreaterThan(Predicate):
    """
    Defines the predicate with greater than or equal to comparison:
    "lhs >= val" where lhs is the signal, and val is the constant.

    Attributes:
        lhs: Left-hand side expression or signal.
        val: Right-hand side constant value for the comparison.

    Methods:
        robustness_value: Computes the weighted robustness of
                          the "greater than" predicate.
        __str__: Returns a string representation of the GreaterThan instance.

    """

    def __init__(self, lhs, val):
        """
        Initialize a GreaterThan instance.

        Args:
            lhs (str or Expression): Left-hand side expression or signal.
            val (float, int, Expression, torch.Tensor): Right-hand side constant value
                                                        for the comparison.
        """
        super(GreaterThan, self).__init__(lhs, val)

    def robustness_value(self, trace, **kwargs):
        """
        Computes the weighted robustness of the "greater than" predicate.

        Args:
            trace (Expression or torch.Tensor): Input signal trace.

        Returns:
            torch.Tensor: Weighted robustness values.
        """
        if isinstance(trace, Expression):
            trace = trace.value
        if isinstance(self.val, Expression):
            return trace - self.val.value
        else:
            return trace - self.val

    def __str__(self):
        """Returns a string representation of the GreaterThan instance."""
        lhs_str = self.lhs
        if isinstance(self.lhs, Expression):
            lhs_str = self.lhs.name
        if isinstance(self.val, str):
            return lhs_str + " >= " + self.val
        if isinstance(self.val, Expression):
            return lhs_str + " >= " + self.val.name
        if isinstance(self.val, torch.Tensor):
            return lhs_str + " >= " + tensor2str(self.val)
        return lhs_str + " >= " + str(self.val)


class Equal(Predicate):
    """
    Defines the predicate with equality comparison:
    "lhs == val" where lhs is the signal, and val is the constant.

    Attributes:
        lhs: Left-hand side expression or signal.
        val: Right-hand side constant value for the comparison.

    Methods:
        robustness_value: Computes the weighted robustness of the "equality" predicate.
        __str__: Returns a string representation of the GreaterThan instance.

    """

    def __init__(self, lhs, val):
        """
        Initialize an Equality instance.

        Args:
            lhs (str or Expression): Left-hand side expression or signal.
            val (float, int, Expression, torch.Tensor): Right-hand side constant value
                                                        for the comparison.
        """
        super(Equal, self).__init__(lhs, val)

    def robustness_value(self, trace, **kwargs):
        """
        Computes the weighted robustness of the "equality" predicate.

        Args:
            trace (Expression or torch.Tensor): Input signal trace.

        Returns:
            torch.Tensor: Weighted robustness values.
        """

        if isinstance(trace, Expression):
            trace = trace.value
        if isinstance(self.val, Expression):
            return -torch.abs(trace - self.val.value)
        else:
            return -torch.abs(trace - self.val)

    def __str__(self):
        """Returns a string representation of the Equality instance."""
        lhs_str = self.lhs
        if isinstance(self.lhs, Expression):
            lhs_str = self.lhs.name

        if isinstance(self.val, str):
            return lhs_str + " = " + self.val
        if isinstance(self.val, Expression):
            return lhs_str + " = " + self.val.name
        if isinstance(self.val, torch.Tensor):
            return lhs_str + " = " + tensor2str(self.val)
        return lhs_str + " = " + str(self.val)


class Negation(WSTL_Formula):
    """
    Defines the negation "¬φ".

    Attributes:
        subformula: Subformula to be negated.

    Methods:
        weight_assignment: Assigns weights to the negation along with its subformula.
        compute_weights: Assigns weights to the negation.
                         As there is no weight in negation, this function returns nothing.
        weight_update: Updates weights in the negation.
                       As there is no weight in negation, this function returns nothing.
        robustness_value: Computes the weighted robustness of the negation.
        __str__: Returns a string representation of the Negation.
    """

    def __init__(self, subformula):
        """
        Initialize a Negation instance.

        Args:
            subformula: Subformula to be negated.
        """
        super(Negation, self).__init__()
        self.subformula = subformula

    def weight_assignment(self, inputs, w_range, no_samples, random, **kwargs):
        """
        Assigns weights to the negation along with its subformula.

        Args:
            inputs (Expression or torch.Tensor or tuple): Input signals.
            w_range (tuple): Weight range for random initialization.
            no_samples (int): Number of weight valuation samples to be set
                              (useful for random case).
            random (bool): Flag for random initialization.
            **kwargs: Additional keyword arguments
        """

        self.subformula.weight_assignment(inputs, w_range, no_samples, random, **kwargs)
        self.compute_weights()

    def compute_weights(self):
        """
        Assigns weights to the negation.

        As there is no weight in negation, this function returns nothing.
        """
        pass

    def weight_update(self):
        """
        Updates weights in the negation.

        As there is no weight in negation, this function returns nothing.
        """
        pass

    def robustness_value(self, inputs, **kwargs):
        """
        Computes the weighted robustness of the negation.

        Args:
            inputs (Expression or torch.Tensor or tuple): Input signals.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Weighted robustness values.
        """

        sc = kwargs["scale"]
        return -self.subformula(inputs, scale=sc)

    def __str__(self):
        return "¬(" + str(self.subformula) + ")"


class And(Logic_Operator):
    """
    Defines "And" operator. And operator needs two subformulas.

    Attributes:
        subformula1: First subformula.
        subformula2: Second subformula.

    Attributes:
        operation: Minish() - The operation used for conjunction.

    Methods:
        compute_weights: Assigns weight values for And.
        compute_robustness: Computes the weighted robustness of And.
        __str__: Returns a string representation of the And.

    """

    def __init__(self, subformula1, subformula2):
        """
        Initializes an instance of the And operator.

        Args:
            subformula1 (WSTL_Formula): First subformula.
            subformula2 (WSTL_Formula): Second subformula.
        """

        super(And, self).__init__(subformula1=subformula1, subformula2=subformula2)
        self.operation = Minish()

    def compute_weights(self, w_range, no_samples, random, **kwargs):
        """
        Assigns weight values for And.

        Args:
            w_range (tuple): A tuple specifying the range for weight assignment.
            no_samples (int): The number of weight samples to generate.
            random (bool): If True, generates random weights;
                           otherwise, uses default weights.
            **kwargs: Additional keyword arguments.
        """

        if random:
            self.weights[
                "("
                + str(self.subformula1).replace(".", "")
                + ") ∧ ("
                + str(self.subformula2).replace(".", "")
                + ")"
            ] = torch.nn.Parameter(
                w_range[0]
                + (w_range[1] - w_range[0])
                * torch.rand(size=(2, no_samples), dtype=torch.float, requires_grad=True)
            )
        else:
            self.weights[
                "("
                + str(self.subformula1).replace(".", "")
                + ") ∧ ("
                + str(self.subformula2).replace(".", "")
                + ")"
            ] = torch.nn.Parameter(
                torch.ones(size=(2, no_samples), dtype=torch.float, requires_grad=True)
            )

    def compute_robustness(self, input_, **kwargs):
        """
        Computes the weighted robustness of And.

        Args:
            input_ (torch.Tensor): The input tensor for robustness computation.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The computed weighted robustness values
        """

        try:
            sc = kwargs["scale"]
        except KeyError:
            sc = -1

        # input_ = input_ / torch.tensor(
        #     [torch.max(input_[:, :, :, 0]), torch.max(input_[:, :, :, 1])]
        # ).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        output_ = torch.Tensor()
        for i in range(input_.shape[1]):
            output_ = torch.cat(
                (
                    output_,
                    self.operation(
                        self.weights[
                            "("
                            + str(self.subformula1).replace(".", "")
                            + ") ∧ ("
                            + str(self.subformula2).replace(".", "")
                            + ")"
                        ].T
                        * input_[:, i, :, :].unsqueeze(1),
                        sc,
                        axis=-1,
                    ),
                ),
                axis=1,
            )
        return output_

    def __str__(self):
        """Returns a string representation of the And."""

        return "(" + str(self.subformula1) + ") ∧ (" + str(self.subformula2) + ")"


class Or(Logic_Operator):
    """
    Defines Or operator. Or operator needs two subformulas.

    Attributes:
        subformula1: First subformula.
        subformula2: Second subformula.

    Methods:
        compute_weights: Assigns weight values for Or.
        compute_robustness: Computes the weighted robustness of Or.
        __str__: Returns a string representation of the Or.

    """

    def __init__(self, subformula1, subformula2):
        """
        Initializes an instance of the Or class.

        Args:
            subformula1 (WSTL_Formula): The first subformula for the "Or" operator.
            subformula2 (WSTL_Formula): The second subformula for the "Or" operator.
        """

        super(Or, self).__init__(subformula1=subformula1, subformula2=subformula2)
        self.operation = Maxish()

    def compute_weights(self, w_range, no_samples, random, **kwargs):
        """
        Assigns weight values for Or.

        Args:
            w_range (tuple): A tuple specifying the range for weight assignment.
            no_samples (int): The number of weight samples to generate.
            random (bool): If True, generates random weights;
                           otherwise, uses default weights.
            **kwargs: Additional keyword arguments.
        """

        if random:
            self.weights[
                "("
                + str(self.subformula1).replace(".", "")
                + ") ∨ ("
                + str(self.subformula2).replace(".", "")
                + ")"
            ] = torch.nn.Parameter(
                w_range[0]
                + (w_range[1] - w_range[0])
                * torch.rand(size=(2, no_samples), dtype=torch.float, requires_grad=True)
            )
        else:
            self.weights[
                "("
                + str(self.subformula1).replace(".", "")
                + ") ∨ ("
                + str(self.subformula2).replace(".", "")
                + ")"
            ] = torch.nn.Parameter(
                torch.ones(size=(2, no_samples), dtype=torch.float, requires_grad=True)
            )

    def compute_robustness(self, input_, **kwargs):
        """
        Computes the weighted robustness of Or.

        Args:
            input_ (torch.Tensor): The input tensor for robustness computation.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The computed weighted robustness values.
        """

        if self.operation is None:
            raise Exception()
        try:
            sc = kwargs["scale"]
        except KeyError:
            sc = -1

        output_ = torch.Tensor()
        for i in range(input_.shape[1]):
            output_ = torch.cat(
                (
                    output_,
                    self.operation(
                        self.weights[
                            "("
                            + str(self.subformula1).replace(".", "")
                            + ") ∨ ("
                            + str(self.subformula2).replace(".", "")
                            + ")"
                        ]
                        * input_[:, i, :],
                        sc,
                    )[:, None, :],
                ),
                axis=1,
            )
        return output_

    def __str__(self):
        return "(" + str(self.subformula1) + ") ∨ (" + str(self.subformula2) + ")"


class Always(Temporal_Operator):
    """
    Defines Always operator. Always operator needs one subformula and an interval.
    If interval is not defined then it is accepted as [0,inf).

    Attributes:
        subformula (WSTL_Formula): The subformula associated with the Always operator.
        interval (Interval): The interval over which the Always operator is evaluated.
        operation (Minish): The operation used for computation.

    Methods:
        compute_weights: Assigns weight values for the Always operator.
        compute_robustness: Computes the weighted robustness of the Always operator.
    """

    def __init__(self, subformula, interval=None):
        super(Always, self).__init__(subformula=subformula, interval=interval)
        self.operation = Minish()

    def compute_weights(self, w_range, no_samples, random):
        """
        Assigns weight values for Always.

         Args:
            w_range (tuple): The range within which weight values are generated.
            no_samples (int): The number of samples for weight values.
            random (bool): If True, generates random weight values;
                           otherwise, uses 1 weights.

        """

        interval_length = self.interval.value[1] - self.interval.value[0] + 1
        if random:
            self.weights[
                "◻ "
                + str(self.interval).replace(".", "")
                + "( "
                + str(self.subformula).replace(".", "")
                + " )"
            ] = torch.nn.Parameter(
                w_range[0]
                + (w_range[1] - w_range[0])
                * torch.rand(
                    size=(interval_length, no_samples),
                    dtype=torch.float,
                    requires_grad=True,
                )
            )
        else:
            self.weights[
                "◻ "
                + str(self.interval).replace(".", "")
                + "( "
                + str(self.subformula).replace(".", "")
                + " )"
            ] = torch.nn.Parameter(
                torch.ones(
                    size=(interval_length, no_samples),
                    dtype=torch.float,
                    requires_grad=True,
                )
            )

    def compute_robustness(self, trace, **kwargs):
        """
        Computes the weighted robustness of Always.

        Args:
            trace (torch.Tensor): The input trace for computing robustness.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The computed weighted robustness.

        """

        try:
            sc = kwargs["scale"]
        except KeyError:
            sc = -1

        w = self.weights[
            "◻ "
            + str(self.interval).replace(".", "")
            + "( "
            + str(self.subformula).replace(".", "")
            + " )"
        ]
        _output = torch.Tensor()
        for i in range(trace.shape[1] - self.interval.value[0]):
            trace_lb = i + self.interval.value[0]
            trace_ub = min(i + self.interval.value[1], trace.shape[1])
            _output = torch.cat(
                (
                    _output,
                    self.operation(
                        w[: min(w.shape[0], trace_ub - trace_lb), :]
                        * trace[:, trace_lb:trace_ub, :, 0],
                        sc,
                        axis=1,
                    ).unsqueeze(-1),
                ),
                axis=1,
            )
        return _output

    def __str__(self):
        """Returns a string representation of the Always operator."""
        return "◻ " + str(self.interval) + "( " + str(self.subformula) + " )"


class Eventually(Temporal_Operator):
    """
    Defines Eventually operator. Eventually operator needs one subformula and an interval.
    If interval is not defined then it is accepted as [0,inf).

    Attributes:
        subformula (WSTL_Formula): The subformula associated with the Eventually operator.
        interval (Interval): The interval over which the Eventually operator is evaluated.
        operation (Maxish): The operation used for computation.

    Methods:
        compute_weights: Assigns weight values for the Eventually operator.
        compute_robustness: Computes the weighted robustness of the Eventually operator.

    """

    def __init__(self, subformula, interval=None):
        super(Eventually, self).__init__(subformula=subformula, interval=interval)
        self.operation = Maxish()

    def compute_weights(self, w_range, no_samples, random):
        """
        Assigns weight values for Eventually.

        Args:
            w_range (tuple): The range within which weight values are generated.
            no_samples (int): The number of samples for weight values.
            random (bool): If True, generates random weight values;
                           otherwise, uses 1 weights.
        """

        interval_length = self.interval.value[1] - self.interval.value[0] + 1
        if random:
            self.weights[
                "♢ "
                + str(self.interval).replace(".", "")
                + "( "
                + str(self.subformula).replace(".", "")
                + " )"
            ] = torch.nn.Parameter(
                w_range[0]
                + (w_range[1] - w_range[0])
                * torch.rand(
                    size=(interval_length, no_samples),
                    dtype=torch.float,
                    requires_grad=True,
                )
            )
        else:
            self.weights[
                "♢ "
                + str(self.interval).replace(".", "")
                + "( "
                + str(self.subformula).replace(".", "")
                + " )"
            ] = torch.nn.Parameter(
                torch.ones(
                    size=(interval_length, no_samples),
                    dtype=torch.float,
                    requires_grad=True,
                )
            )

    def compute_robustness(self, trace, **kwargs):
        """
        Computes the weighted robustness of Eventually.

        Args:
            trace (torch.Tensor): The input trace for computing robustness.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The computed weighted robustness.
        """

        if self.operation is None:
            raise Exception()
        try:
            sc = kwargs["scale"]
        except KeyError:
            sc = -1

        w = self.weights[
            "♢ "
            + str(self.interval).replace(".", "")
            + "( "
            + str(self.subformula).replace(".", "")
            + " )"
        ]
        _output = torch.Tensor()
        for i in range(trace.shape[1] - self.interval.value[0]):
            trace_lb = i + self.interval.value[0]
            trace_ub = min(i + self.interval.value[1], trace.shape[1])
            _output = torch.cat(
                (
                    _output,
                    self.operation(
                        w[: min(w.shape[0], trace_ub - trace_lb), :]
                        * trace[:, trace_lb:trace_ub, :, 0],
                        sc,
                        axis=1,
                    ).unsqueeze(-1),
                ),
                axis=1,
            )
        return _output

    def __str__(self):
        return "♢ " + str(self.interval) + "( " + str(self.subformula) + " )"


class Until(WSTL_Formula):
    """
    Defines Until operator. Until needs two subformulas and an interval.
    If interval is not defined then it is accepted as [0,inf).

    Attributes:
        subformula1 (WSTL_Formula): The first subformula
        subformula2 (WSTL_Formula): The second subformula.
        interval (Interval): The interval over which the operator is evaluated.

    Methods:
        weight_assignment: Assigns weight values for the Until operator.
        weight_update: Updates weights in the subformulas.
        robustness_value: Computes the robustness value for given inputs.
        compute_weights: Assigns weight values for the Until operator.
        compute_robustness: Computes the robustness for a given trace.

    """

    def __init__(self, subformula1, subformula2, interval=None):
        super(Until, self).__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2
        self.interval = Interval(interval)

        assert isinstance(
            self.subformula1, WSTL_Formula
        ), "Subformula1 needs to be an STL formula."
        assert isinstance(
            self.subformula2, WSTL_Formula
        ), "Subformula2 needs to be an STL formula."

    def weight_assignment(self, inputs, w_range, no_samples, random, **kwargs):
        """
        Assigns weight values for the Until operator.

        Args:
            inputs (tuple): Input signals for the subformulas.
            w_range (tuple): The range within which weight values are generated.
            no_samples (int): The number of samples for weight values.
            random (bool): If True, generates random weight values;
                           otherwise, uses 1 weights.
            **kwargs: Additional keyword arguments.
        """

        try:
            sc = kwargs["scale"]
        except KeyError:
            sc = -1

        # self.compute_weights(w_range, no_samples, random)

        self.subformula1.weight_assignment(
            inputs[0], w_range, no_samples, random, **kwargs
        )
        self.subformula2.weight_assignment(
            inputs[1], w_range, no_samples, random, **kwargs
        )
        for keys in self.subformula1.weights.keys():
            self.weights[keys] = self.subformula1.weights[keys]
        for keys in self.subformula2.weights.keys():
            self.weights[keys] = self.subformula2.weights[keys]

        trace1 = self.subformula1(inputs[0], scale=sc)
        trace2 = self.subformula2(inputs[1], scale=sc)
        self.interval.set_interval(min(trace1.shape[1], trace2.shape[1]))

        return self.compute_weights(w_range, no_samples, random)

    def weight_update(self):
        """Updates weights in the subformulas based on the Until operator's weights."""
        for keys in self.subformula1.weights.keys():
            self.subformula1.weights[keys] = self.weights[keys]
        for keys in self.subformula2.weights.keys():
            self.subformula2.weights[keys] = self.weights[keys]
        self.subformula1.weight_update()
        self.subformula2.weight_update()

    def robustness_value(self, inputs, **kwargs):
        """
        Computes the robustness value of the Until operator for given inputs.

        Args:
            inputs (tuple): Input signals for the subformulas.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The computed robustness value.

        """
        try:
            sc = kwargs["scale"]
        except KeyError:
            sc = -1
        trace1 = self.subformula1(inputs[0], scale=sc)
        trace2 = self.subformula2(inputs[1], scale=sc)

        trace_length = min(trace1.shape[1], trace2.shape[1])
        trace = torch.cat(
            [trace1[:, :trace_length, :], trace2[:, :trace_length, :]], axis=-1
        )

        return self.compute_robustness(trace, scale=sc)

    def compute_weights(self, w_range, no_samples, random):
        """
        Assigns weight values for the Until operator.

        Args:
            w_range (tuple): The range within which weight values are generated.
            no_samples (int): The number of samples for weight values.
            random (bool): If True, generates random weight values;
                        otherwise, uses uniform weights.

        """
        interval_length = self.interval.value[1] - self.interval.value[0] + 1
        if random:
            self.weights[
                "("
                + str(self.subformula1).replace(".", "")
                + ")"
                + " U "
                + "("
                + str(self.subformula2).replace(".", "")
                + ")"
            ] = torch.nn.Parameter(
                w_range[0]
                + (w_range[1] - w_range[0])
                * torch.rand(
                    size=(2, interval_length, no_samples),
                    dtype=torch.float,
                    requires_grad=True,
                )
            )
        else:
            self.weights[
                "("
                + str(self.subformula1).replace(".", "")
                + ")"
                + " U "
                + "("
                + str(self.subformula2).replace(".", "")
                + ")"
            ] = torch.nn.Parameter(
                torch.ones(
                    size=(2, interval_length, no_samples),
                    dtype=torch.float,
                    requires_grad=True,
                )
            )

    def compute_robustness(self, trace, scale=-1):
        """
        Computes the weighted robustness of the Until operator for a given trace.

        Args:
            trace (torch.Tensor): The input trace for computing robustness.
            scale (float): Scaling factor for the computation.

        Returns:
            torch.Tensor: The computed robustness.

        """
        w = self.weights[
            "("
            + str(self.subformula1).replace(".", "")
            + ")"
            + " U "
            + "("
            + str(self.subformula2).replace(".", "")
            + ")"
        ]

        _output = torch.Tensor()

        mins = Minish()
        maxs = Maxish()

        for i in range(trace.shape[1] - self.interval.value[0]):
            trace_lb = i + self.interval.value[0]
            internal_trace = torch.Tensor()
            for k in range(trace_lb, min(trace.shape[1], i + self.interval.value[1])):
                internal_min = mins(trace[:, trace_lb : k + 1, :, 0], scale, axis=1)
                min_compare = torch.cat(
                    (
                        trace[:, k, :, 1].reshape(trace.shape[0], 1, trace.shape[2]),
                        internal_min.reshape(internal_min.shape[0], 1, trace.shape[2]),
                    ),
                    axis=1,
                )
                internal_trace = torch.cat(
                    (
                        internal_trace,
                        mins(w[:, k - trace_lb, :] * min_compare, scale, axis=1),
                    ),
                    axis=1,
                )
            _output = torch.cat(
                (_output, maxs(internal_trace, scale, axis=1).unsqueeze(-1)), axis=1
            )

        return _output

    def __str__(self):
        if self.interval is None:
            int = [0, np.inf]
        else:
            int = self.interval
        return (
            "( "
            + str(self.subformula1)
            + " )"
            + "U"
            + str(int)
            + "( "
            + str(self.subformula2)
            + " )"
        )


class Expression(torch.nn.Module):
    """
    Wraps a pytorch arithmetic operation, so that we can intercept and overload comparison
    operators. Expression allows us to express tensors using their names to make it easier
    to code up and read, but also keep track of their numeric values.

    Attributes:
        name (str): The name associated with the expression.
        value (float or torch.Tensor): The numeric value of the expression.

    Methods:
        set_name(new_name): Updates the name attribute of the expression.
        set_value(new_value): Updates the value attribute of the expression.

    """

    def __init__(self, name, value):
        super(Expression, self).__init__()
        self.name = name
        self.value = value

    def set_name(self, new_name):
        """
        Updates the name attribute of the expression.

        Args:
            new_name (str): The new name to be assigned to the expression.
        """
        self.name = new_name

    def set_value(self, new_value):
        """
        Updates the value attribute of the expression.

        Args:
            new_value (float or torch.Tensor): The new numeric value to be assigned.
        """
        self.value = new_value

    def __neg__(self):
        return Expression(-self.value)

    def __add__(self, other):
        if isinstance(other, Expression):
            return Expression(self.name + "+" + other.name, self.value + other.value)
        else:
            return Expression(self.name + "+other", self.value + other)

    def __radd__(self, other):
        return self.__add__(other)
        # No need for the case when "other" is an Expression, since that
        # case will be handled by the regular add

    def __sub__(self, other):
        if isinstance(other, Expression):
            return Expression(self.name + "-" + other.name, self.value - other.value)
        else:
            return Expression(self.name + "-other", self.value - other)

    def __rsub__(self, other):
        return Expression(other - self.value)
        # No need for the case when "other" is an Expression, since that
        # case will be handled by the regular sub

    def __mul__(self, other):
        if isinstance(other, Expression):
            return Expression(self.name + "*" + other.name, self.value * other.value)
        else:
            return Expression(self.name + "*other", self.value * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(a, b):
        # This is the new form required by Python 3
        numerator = a
        denominator = b
        num_name = "num"
        denom_name = "denom"
        if isinstance(numerator, Expression):
            num_name = numerator.name
            numerator = numerator.value
        if isinstance(denominator, Expression):
            denom_name = denominator.name
            denominator = denominator.value
        return Expression(num_name + "/" + denom_name, numerator / denominator)

    # Comparators
    def __le__(lhs, rhs):
        assert isinstance(lhs, str) | isinstance(
            lhs, Expression
        ), "LHS of LessThan needs to be a string or Expression"
        assert not isinstance(rhs, str), "RHS cannot be a string"
        return LessThan(lhs, rhs)

    def __ge__(lhs, rhs):
        assert isinstance(lhs, str) | isinstance(
            lhs, Expression
        ), "LHS of GreaterThan needs to be a string or Expression"
        assert not isinstance(rhs, str), "RHS cannot be a string"
        return GreaterThan(lhs, rhs)

    def __eq__(lhs, rhs):
        assert isinstance(lhs, str) | isinstance(
            lhs, Expression
        ), "LHS of Equal needs to be a string or Expression"
        assert not isinstance(rhs, str), "RHS cannot be a string"
        return Equal(lhs, rhs)

    def __str__(self):
        return str(self.name)


def get_input_length(inputs):
    """
    Turns a tensor into a string for printing.

    Args:
        tensor (torch.Tensor): The input PyTorch tensor.

    Returns:
        str: A string representation of the tensor for printing.

    """
    if isinstance(inputs, tuple):
        inputs = inputs[0]
        signal_no = get_input_length(inputs)
    elif isinstance(inputs, torch.Tensor):
        signal_no = inputs.shape[0]
    else:
        raise TypeError("signals are not in the correct form.")

    return signal_no


def tensor2str(tensor):
    """
    Turns a tensor into a string for printing
    """
    device = tensor.device.type
    tensor = tensor.detach()
    if device == "cuda":
        tensor = tensor.cpu()
    return str(tensor.numpy())


def convert_inputs(inputs):
    """
    Parses the tuple of signals into signals for subformulas.

    This function converts the input signals for subformulas, which may include
    Expressions, torch Tensors, or tuples of signals.

    Args:
        inputs (tuple): A tuple of signals to be converted.

    Returns:
        tuple: A tuple of converted signals for subformulas.
    """
    _x, _y = inputs
    if isinstance(_x, Expression):
        if _x.value is None:
            raise ValueError("Input Expression does not have numerical values")
        x_return = _x.value
    elif isinstance(_x, torch.Tensor):
        x_return = _x
    elif isinstance(_x, tuple):
        x_return = convert_inputs(_x)
    else:
        raise ValueError("First argument is an invalid input trace")

    if isinstance(_y, Expression):
        if _y.value is None:
            raise ValueError("Input Expression does not have numerical values")
        y_return = _y.value
    elif isinstance(_y, torch.Tensor):
        y_return = _y
    elif isinstance(_y, tuple):
        y_return = convert_inputs(_y)
    else:
        raise ValueError("Second argument is an invalid input trace")

    return (x_return, y_return)


class Interval:
    """
    Defines intervals for temporal operators.

    Attributes:
        interval: The interval specified during initialization.
        value: The computed interval values based on the input length.

    Methods:
        set_interval: Sets the interval for robustness computation.
    """

    def __init__(self, interval):
        """
        Initializes an instance of the Interval class.

        Args:
            interval (tuple): A tuple specifying the interval boundaries.
        """
        self.interval = interval
        pass

    def set_interval(self, input_length):
        """
        Sets the interval for robustness computation.
        If interval goes to infinity, interval for the computation
        needs to be full signal length.

        Args:
            input_length (int): The length of the input signal.

        """
        if self.interval is None:
            self.value = [0, input_length - 1]
        elif self.interval[1] == np.inf:
            self.value = [self.interval[0], input_length - 1]
        else:
            self.value = self.interval
        return

    def __str__(self):
        """Returns a string representation of the Interval."""
        if self.interval is None:
            return ""
        elif self.interval[1] == np.inf:
            return f"[{str(self.interval[0])}, ∞]"
        else:
            return str(self.interval)
