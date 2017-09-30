package net.demilich.metastone.game.behaviour.features;

import net.demilich.metastone.game.Player;
import net.demilich.metastone.game.behaviour.GreedyOptimizeMoveModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

public class Feature_basic {

    private int hash_code;
    private int winner = -1;
    private List<List<Integer>> states = new ArrayList<>();
    private String fea_name;
    private final static Logger logger = LoggerFactory.getLogger(GreedyOptimizeMoveModel.class);
    private int featureSingleLen = 0;

    public Feature_basic(int hash, String fea_name){
        this.hash_code = hash;
        this.fea_name = fea_name;
    }

    public int feature_number(){
        return states.get(0).size();
    }

    public void appendWrite(String filename){
        FileWriter fw = null;
        assert(this.winner != -1);
        try {
            //如果文件存在，则追加内容；如果文件不存在，则创建文件
            File f=new File(filename);
            fw = new FileWriter(f, true);
        } catch (IOException e) {
            e.printStackTrace();
        }
        PrintWriter pw = new PrintWriter(fw);
        pw.println(this.toString());
        pw.flush();
        try {
            fw.flush();
            pw.close();
            fw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void append(Player[] players, int activePlayer){
        List<Integer> envState = new ArrayList<>();
        if (this.fea_name.equals("Basic")){
            envState.add(activePlayer);
            envState.addAll(players[activePlayer].getPlayerStateBasic());
            featureSingleLen = envState.size();
            envState.addAll(players[1-activePlayer].getPlayerStateBasic());
        }
        else if (this.fea_name.equals("feature_fh_0")){
//            envState.add(activePlayer);
//            envState.addAll(players[activePlayer].getPlayerStatefh0(false));
//            featureSingleLen = envState.size();
//            envState.addAll(players[1-activePlayer].getPlayerStatefh0(true));
        }
        else if (this.fea_name.equals("feature_fh_1")){
            envState.add(activePlayer);
            envState.addAll(players[0].getPlayerStatefh1(false));
            featureSingleLen = envState.size();
            envState.addAll(players[1].getPlayerStatefh1(true));
        }
        else{
            logger.info("No known feature");
        }

        states.add(envState);
        //logger.info("envState Size: {}, states Size: {}", envState.size(), states.size());
    }

    public void end(int winner_id){
        this.winner = winner_id;
    }

    @Override
    public String toString(){
        StringBuilder sb = new StringBuilder();
        for (int turn=0; turn < this.states.size(); turn++){
            sb.append("{'GameHash':" + this.hash_code + ",'Turn':" + (turn+1));
            sb.append(",'Active':'" + this.states.get(turn).get(0) + "','playerActive':'");
            int size = this.states.get(turn).size();
//            logger.info("Feature Size: {}", size);

            for (int fe=1; fe<size; fe++){
                if (fe == 1 || fe == featureSingleLen)
                    sb.append(this.states.get(turn).get(fe));
                else
                    sb.append("|" + this.states.get(turn).get(fe));

                if (fe == featureSingleLen - 1){
                    sb.append("'");
                    sb.append(",'playerOpposite':'");
                }
            }
            sb.append("'" + ",'winner':" + this.winner + "}\n");
        }
//        sb.append("{'GameHash':" + this.hash_code + ",'Turn':" + this.states.size() + ",'winner':" + this.winner + "}");
        return sb.toString();
    }
}
