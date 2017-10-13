package net.demilich.metastone.game.behaviour;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

public class LookAndModify {
    private final static Logger logger = LoggerFactory.getLogger(LookAndModify.class);

    public static void main(String[] args) {
        INDArray para;
        try{
            File readFile = new File("app/NdModel/linear/96feaGameTree_mean_para_ES_turnEnd.data");
            para = Nd4j.readBinary(readFile);
            INDArray w = para.get(NDArrayIndex.interval(0,96), NDArrayIndex.all()); // interval左闭区间右开区间
            INDArray b = para.get(NDArrayIndex.point(96), NDArrayIndex.all());
            logger.info("W: {}", w);
            logger.info("b: {}", b);
            w.putScalar(new int[]{0, 0}, 1); // 改变指定位置上的值
            logger.info("W: {}", w);
            para = Nd4j.concat(0, w, b);
            logger.info("Total: {}", para);
            // 保存到本地
//            File saveFile = new File(saveFileName);
//            Nd4j.saveBinary(para, saveFile);
        }
        catch (IOException e){
            e.printStackTrace();
        }
    }
}
