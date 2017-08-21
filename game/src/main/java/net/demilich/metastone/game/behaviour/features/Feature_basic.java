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

    public void append(Player[] players){
        List<Integer> envState = new ArrayList<>();
        if (this.fea_name.equals("Basic")){
            envState.addAll(players[0].getPlayerStateBasic());
            envState.addAll(players[1].getPlayerStateBasic());
        }
        else if (this.fea_name.equals("feature_fh_0")){
            envState.addAll(players[0].getPlayerStatefh0());
            envState.addAll(players[1].getPlayerStatefh0());
        }
        else if (this.fea_name.equals("feature_fh_1")){
            envState.addAll(players[0].getPlayerStatefh1());
            envState.addAll(players[1].getPlayerStatefh1());
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
            sb.append(",'player" + 0 + "':'");
            int size = this.states.get(turn).size();
            for (int fe=0; fe<size; fe++){
                if (fe == 0 || fe == size / 2)
                    sb.append(this.states.get(turn).get(fe));
                else
                    sb.append("|" + this.states.get(turn).get(fe));

                if (fe == size / 2 - 1){
                    sb.append("'");
                    sb.append(",'player" + 1 + "':'");
                }
            }
            sb.append("'}\n");
        }
        sb.append("{'GameHash':" + this.hash_code + ",'Turn':" + this.states.size() + ",'winner':" + this.winner + "}");
        return sb.toString();
    }
}
