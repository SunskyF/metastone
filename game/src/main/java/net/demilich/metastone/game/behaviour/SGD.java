package net.demilich.metastone.game.behaviour;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SGD{
    private final static Logger logger = LoggerFactory.getLogger(SGD.class);

    double[] v;
    double lr;
    double momentum;
    public SGD(int feaNum, double lr, double momentum){
        this.lr = lr;
        this.momentum = momentum;
        this.v = new double[feaNum];
    }

    public double[] get_gradient(double[] grad){
        assert (grad.length == v.length);
        double[] res = new double[this.v.length];
        for(int i = 0; i < this.v.length; i++){
            this.v[i] = this.momentum * this.v[i] + (1.0 - this.momentum) * grad[i];
            res[i] = this.v[i] * this.lr;
        }
        return res;
    }
}
