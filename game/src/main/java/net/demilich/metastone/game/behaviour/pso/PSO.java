package net.demilich.metastone.game.behaviour.pso;

import net.demilich.metastone.game.behaviour.LinearBatchPSO;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class PSO
{
    private final static Logger logger = LoggerFactory.getLogger(PSO.class);

    public PSO()
    {
        agent = new Agent[Agent.iPOSNum];
        for(int i =0;i < Agent.iPOSNum;i++){
            agent[i] = new Agent();
        }
    }

    public void update(){
        for (int i =0;i < Agent.iPOSNum;++i){
            agent[i].UpdatePos();
        }
    }

    public Agent[] agent;
}
