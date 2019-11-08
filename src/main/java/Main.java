import java.io.BufferedReader;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

public class Main {

    public static int nextInt(BufferedReader sc) throws IOException {
        int a = 0;
        int k = 1;
        int b = sc.read();
        while (b < '0' || b > '9') {
            if (b == '-') k = -1;
            b = sc.read();
        }
        while (b >= '0' && b <= '9') {
            a = a * 10 + (b - '0');
            b = sc.read();
        }
        return a * k;
    }

    public static void main(String[] args) throws IOException {
        BufferedReader r = new BufferedReader(new InputStreamReader(System.in));
        int n = nextInt(r);
        int m = nextInt(r);
        int d = nextInt(r);
        int[] tiles = new int[m];
        int totalSum = 0;
        for (int i = 0; i < m; i++) {
            tiles[i] = nextInt(r);
            totalSum += tiles[i];
        }
        int[] res = new int[n];
        int cur = -1;
        for (int i = 0; i < m; i++) {
            if (cur + d + totalSum >= n) {
                int jump = n - totalSum - cur;
                cur += jump;
                for (int j = i; j < m; j++) {
                    for (int k = 0; k < tiles[j]; k++) {
                        res[cur] = j + 1;
                        cur++;
                    }
                }
                System.out.println("YES");
                for (int z: res) System.out.print(z + " ");
                return;
            }
            cur += d;
            for (int k = 0; k < tiles[i]; k++) {
                res[cur] = i + 1;
                cur++;
            }
            cur--;
            totalSum -= tiles[i];
        }
        if (cur + d >= n) {
            System.out.println("YES");
            for (int z: res) System.out.print(z + " ");
        } else {
            System.out.println("NO");
        }
    }
}