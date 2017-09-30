package net.demilich.metastone.game.behaviour.pso;

import java.util.Random;

public class Agent {
    public Agent() //对粒子的位置和速度进行初始化
    {
        for(int i = 0; i < iAgentDim; i++)
        {
            pos[i] = 2 * random.nextDouble() - 1;
            v[i] = posBest[i] = pos[i];
        }
    }
    public void UpdateFitness(double fitness)
    {
        nowFitness = fitness;
        if(nowFitness > bestFitness)
        {
            bestFitness = nowFitness;
            posBest = pos.clone();
        }
        if (nowFitness >= gBestFitness){
            gBestFitness = nowFitness;
            gbest = pos.clone();
        }
    }

    public void UpdatePos()
    {
        for(int i = 0;i < iAgentDim;i++)
        {
            v[i] = w * v[i] + delta1 * random.nextDouble() * (posBest[i] - pos[i]) + delta2 * random.nextDouble() * (gbest[i]   - pos[i]);
            pos[i] += v[i];
        }
        this.epoch++;
        this.w = (wStart - wEnd) * (20 - epoch) / 20 + wEnd; // 最大迭代次数为20
    }
    public static int iPOSNum = 20;
    public static int iAgentDim = 89;

    private double w = 0.9;
    private double wStart = 0.9;
    private double wEnd = 0.4;
    private final double delta1 = 2;
    private final double delta2 = 2;

    public double[] pos = new double[iAgentDim];    //粒子的位置
    public double[] posBest = new double[iAgentDim];  //粒子本身的最优位置
    public double[] v = new double[iAgentDim];      //粒子的速度
    public static double[] gbest = new double[iAgentDim];

    public double nowFitness;
    public static double gBestFitness = -1;
    public double bestFitness  = -1;
    public int epoch = 0;
    private Random random = new Random();
}
