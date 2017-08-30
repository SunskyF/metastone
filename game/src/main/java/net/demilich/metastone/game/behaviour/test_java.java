package net.demilich.metastone.game.behaviour;

import java.util.ArrayList;
import java.util.List;

public class test_java {
    public static void main(String[] args) {
        List<Integer> test_0 = new ArrayList<>();
        test_0.add(0);
        test_0.add(1);
        test_0.add(2);
        test_0.add(3);

        List<Integer> test_1 = new ArrayList<>();
        test_1.add(0);
        test_1.add(1);
        test_1.add(2);
        test_1.add(3);

        System.out.println(test_0.hashCode());
        System.out.println(test_1.hashCode());
    }
}
