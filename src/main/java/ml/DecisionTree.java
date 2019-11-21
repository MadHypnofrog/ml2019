package ml;

import javafx.util.Pair;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import static ml.Utils.nextInt;

public class DecisionTree {

    final private static int ENSEMBLE_SIZE = 3000;

    private static class Node {

        int result;
        int splitClass;
        double value;
        Node left, right;

        Node(Node left, Node right, int splitClass, double value) {
            this.left = left;
            this.right = right;
            this.splitClass = splitClass;
            this.value = value;
        }

        Node(int result) {
            this.result = result;
            left = right = null;
        }

    }

    private static Node buildRecursiveTree(int[][] objects, int maxH, int curH, int numClasses,
                                           int features, Set<Integer> skipped) {
        int[] classes = new int[numClasses];
        for (int[] obj : objects) classes[obj[features] - 1]++;
        int maxC = 0;
        for (int i = 0; i < numClasses; i++) {
            if (classes[i] == objects.length) {
                return new Node(i + 1);
            }
            if (classes[maxC] < classes[i]) maxC = i;
        }
        if (curH == maxH) {
            return new Node(maxC + 1);
        }
        int splitClass = -1;
        int splitIndex = -1;
        double value = -1;
        double maxGini = 0;
        for (int i = 0; i < features; i++) {
            if (skipped.contains(i)) continue;
            final int fI = i;
            Arrays.sort(objects, Comparator.comparingInt(a -> a[fI]));
            int[] clLeft = new int[numClasses];
            int[] clRight = new int[numClasses];
            System.arraycopy(classes, 0, clRight, 0, numClasses);
            double gLeft = 0;
            double gRight = 0;
            for (int j = 0; j < numClasses; j++) {
                gRight += clRight[j] * clRight[j];
            }
            gRight /= objects.length;
            for (int objLeft = 0; objLeft < objects.length - 1; objLeft++) {
                if (objects[objLeft][i] == objects[objects.length - 1][i]) break;
                while (objLeft != objects.length - 2 && objects[objLeft][i] == objects[objLeft + 1][i]) {
                    int cl = objects[objLeft][features] - 1;
                    gRight = gRight * (objects.length - objLeft) - clRight[cl] * clRight[cl];
                    clRight[cl]--;
                    gRight = (gRight + clRight[cl] * clRight[cl]) / (objects.length - objLeft - 1);
                    gLeft = gLeft * objLeft - clLeft[cl] * clLeft[cl];
                    clLeft[cl]++;
                    gLeft = (gLeft + clLeft[cl] * clLeft[cl]) / (objLeft + 1);
                    objLeft++;
                }
                int cl = objects[objLeft][features] - 1;
                gRight = gRight * (objects.length - objLeft) - clRight[cl] * clRight[cl];
                clRight[cl]--;
                gRight = (gRight + clRight[cl] * clRight[cl]) / (objects.length - objLeft - 1);
                gLeft = gLeft * objLeft - clLeft[cl] * clLeft[cl];
                clLeft[cl]++;
                gLeft = (gLeft + clLeft[cl] * clLeft[cl]) / (objLeft + 1);
                if (gLeft + gRight > maxGini || gLeft + gRight == maxGini
                        && Math.abs(objects.length / 2 - splitIndex) > Math.abs(objects.length / 2 - objLeft - 1)) {
                    maxGini = gLeft + gRight;
                    splitClass = i + 1;
                    value = (objects[objLeft][i] + objects[objLeft + 1][i]) / 2D;
                    splitIndex = objLeft + 1;
                }
            }
        }
        final int fCl = splitClass - 1;
        Arrays.sort(objects, Comparator.comparingInt(a -> a[fCl]));
        return new Node
                (buildRecursiveTree(Arrays.copyOfRange(objects, 0, splitIndex),
                        maxH, curH + 1, numClasses, features, skipped),
                        buildRecursiveTree(Arrays.copyOfRange(objects, splitIndex, objects.length),
                                maxH, curH + 1, numClasses, features, skipped),
                        splitClass, value);
    }

    private static int classify(int[] object, Node tree) {
        if (tree.left == null) {
            return tree.result;
        }
        if (object[tree.splitClass - 1] < tree.value) return classify(object, tree.left);
        else return classify(object, tree.right);
    }

    private static int classifyEnsemble(int[] object, List<Node> ensemble) {
        Map<Integer, Integer> voting = new HashMap<>();
        for (Node n : ensemble) voting.merge(classify(object, n), 1, (x, y) -> x + y);
        return voting.entrySet().stream().max(Comparator.comparingInt(Map.Entry::getValue)).get().getKey();
    }

    public static void main(String[] args) throws IOException {
        BufferedWriter log = new BufferedWriter(new FileWriter("log-dt.txt"));
        List<Pair<Integer, Integer>> optHeights = new ArrayList<>();
        XYSeriesCollection ds = new XYSeriesCollection();
        XYSeries series = new XYSeries("");
        for (int set = 1; set <= 21; set++) {
            BufferedReader rTrain = new BufferedReader(new FileReader(
                    new File("src/main/resources/dt/" + (set < 10 ? "0" : "") + set + "_train.txt")));
            BufferedReader rTest = new BufferedReader(new FileReader(
                    new File("src/main/resources/dt/" + (set < 10 ? "0" : "") + set + "_test.txt")));

            int features = nextInt(rTrain);
            int numClasses = nextInt(rTrain);
            int nTrain = nextInt(rTrain);
            int[][] train = new int[nTrain][features + 1];
            for (int i = 0; i < nTrain; i++) {
                for (int j = 0; j < features + 1; j++) {
                    train[i][j] = nextInt(rTrain);
                }
            }

            int featTest = nextInt(rTest);
            if (features != featTest) {
                System.out.println("Incorrect number of features in set " + set);
            }
            int numClTest = nextInt(rTest);
            if (numClasses != numClTest) {
                System.out.println("Incorrect number of classes in set " + set);
            }
            int nTest = nextInt(rTest);
            int[][] test = new int[nTest][features + 1];
            for (int i = 0; i < nTest; i++) {
                for (int j = 0; j < features + 1; j++) {
                    test[i][j] = nextInt(rTest);
                }
            }

            XYSeriesCollection dsSet = new XYSeriesCollection();
            XYSeries seriesTrain = new XYSeries("Training set");
            XYSeries seriesTest = new XYSeries("Test set");
            int maxH = 0;
            double maxF = 0;
            double maxFTrain = 0;
            for (int height = 0; height < 50; height++) {
                Node tree = buildRecursiveTree(train, height, 0, numClasses, features, new HashSet<>());
                int[][] conf = new int[numClasses][numClasses];
                for (int i = 0; i < nTest; i++) {
                    conf[classify(test[i], tree) - 1][test[i][features] - 1]++;
                }
                double f1 = Utils.fmeasure(conf);

                int[][] confTrain = new int[numClasses][numClasses];
                for (int i = 0; i < nTrain; i++) {
                    confTrain[classify(train[i], tree) - 1][train[i][features] - 1]++;
                }
                double f1Train = Utils.fmeasure(confTrain);
                log.write(String.format("Set %d, h = %d, f1Test = %f, f1Train = %f\n", set, height, f1, f1Train));
                log.flush();
                if (f1 > maxF) {
                    maxF = f1;
                    maxH = height;
                    maxFTrain = f1Train;
                }
                seriesTrain.add(height, f1Train);
                seriesTest.add(height, f1);
                if (f1Train == 1) break;
            }
            dsSet.addSeries(seriesTrain);
            dsSet.addSeries(seriesTest);
            Utils.writeChartForDS("Set " + set, dsSet, "Height", "F1");
            optHeights.add(new Pair<>(set, maxH));
            series.add(set, maxH);
            log.write(String.format("Optimal height: %d, f1Test = %f, f1Train = %f\n", maxH, maxF, maxFTrain));
            log.flush();

            List<Node> ensemble = new ArrayList<>();
            Random R = new Random();
            for (int i = 0; i < ENSEMBLE_SIZE; i++) {
                int[][] trainObjects = new int[nTrain][];
                for (int j = 0; j < nTrain; j++) {
                    trainObjects[j] = train[R.nextInt(nTrain)];
                }
                HashSet<Integer> skipped = new HashSet<>();
                for (int j = 0; j < features; j++) {
                    skipped.add(j);
                }
                while (skipped.size() != features - (int) Math.sqrt(features)) {
                    skipped.remove(R.nextInt(features));
                }
                ensemble.add(buildRecursiveTree(trainObjects, 200, 0, numClasses, features, skipped));
            }
            int[][] conf = new int[numClasses][numClasses];
            for (int i = 0; i < nTest; i++) {
                conf[classifyEnsemble(test[i], ensemble) - 1][test[i][features] - 1]++;
            }
            double f1 = Utils.fmeasure(conf);
            int[][] confTrain = new int[numClasses][numClasses];
            for (int i = 0; i < nTrain; i++) {
                confTrain[classifyEnsemble(train[i], ensemble) - 1][train[i][features] - 1]++;
            }
            double f1Train = Utils.fmeasure(confTrain);
            log.write(String.format("Using random forest: f1Test = %f, f1Train = %f\n\n", f1, f1Train));
            log.flush();

        }
        ds.addSeries(series);
        Utils.writeChartForDS("Set to optimal height", ds, "Set", "Optimal height");
        optHeights.sort(Comparator.comparingInt(Pair::getValue));
        int minOptHeightSet = optHeights.get(0).getKey();
        int midOptHeightSet = optHeights.get(10).getKey();
        int maxOptHeightSet = optHeights.get(20).getKey();
        for (int set = 1; set <= 21; set++) {
            Path graph = new File("Set " + set + ".png").toPath();
            if (set == minOptHeightSet) {
                Files.move(graph, graph.resolveSibling("min-" + set + ".png"));
            } else if (set == midOptHeightSet) {
                Files.move(graph, graph.resolveSibling("mid-" + set + ".png"));
            } else if (set == maxOptHeightSet) {
                Files.move(graph, graph.resolveSibling("max-" + set + ".png"));
            } else {
                Files.delete(graph);
            }
        }
    }

}
