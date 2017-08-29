package net.demilich.metastone.game.behaviour.mcs;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MultipleExecutor {
    private static final Logger LOG = LoggerFactory.getLogger(MultipleExecutor.class);

    private final int threadNum;
    private final Thread[] threads;

    public MultipleExecutor(int threadNum) {
        this.threadNum = threadNum;
        this.threads = new Thread[threadNum];
    }

    public void execute(Runnable r) {
        if (r == null)
            throw new NullPointerException();

        waitToFinish();

        LOG.debug("Adding new task...");
        for (int i = 0; i < threadNum; i++) {
            threads[i] = new Thread(r);
            threads[i].start();
        }
    }

    public void waitToFinish() {
        LOG.debug("Waiting for the last task to finish...");
        for (int i = 0; i < threadNum; i++) {
            if (threads[i] != null) {
                try {
                    threads[i].join();
                } catch (InterruptedException e) {
                    // Interrupted...
                    LOG.warn("The main thread is interrupted. Interrupting all worker threads...");
                    interrupt();
                    return;
                }
            }
        }
        LOG.debug("The last task is finished.");
    }

    public void interrupt() {
        for (int i = 0; i < threadNum; i++) {
            if (threads[i] != null)
                threads[i].interrupt();
        }
    }

}
