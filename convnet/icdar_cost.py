from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin

class LogCost(DefaultDataSpecsMixin, Cost):
    """
    The default Cost to use with an MLP.

    It simply calls the MLP's cost_from_X method.
    """

    supervised = True

    def expr(self, model, data, **kwargs):
        """
        Parameters
        ----------
        model : MLP
        data : tuple
            Should be a valid occupant of
            CompositeSpace(model.get_input_space(),
            model.get_output_space())

        Returns
        -------
        rval : theano.gof.Variable
            The cost obtained by calling model.cost_from_X(data)
        """
        space, sources = self.get_data_specs(model)
        space.validate(data)

        model.cost_from_X_data_specs()[0].validate(data)
        X, Y = data
        Y_hat = model.fprop(X)

        print("Teeeest: ", Y.shape, Y_hat.shape)
        return model.cost_from_X(data)