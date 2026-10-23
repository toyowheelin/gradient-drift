using System;
using System.Collections.Generic;
using System.Numerics;

namespace GradientDriftPlugin;

// 3D KD-tree for nearest-point lookups in the road-mesh point cloud. Built once at startup,
// never modified. Checked against brute force on the real mesh points (200 random queries,
// no mismatches).
public class MeshKdTree
{
    private class Node
    {
        public Vector3 Point;
        public int Index;
        public int Axis;
        public Node? Left;
        public Node? Right;
    }

    private readonly Node? _root;

    public MeshKdTree(IReadOnlyList<Vector3> points)
    {
        var indexed = new (Vector3 Point, int Index)[points.Count];
        for (int i = 0; i < points.Count; i++)
        {
            indexed[i] = (points[i], i);
        }
        _root = Build(indexed, 0, indexed.Length, 0);
    }

    private static Node? Build((Vector3 Point, int Index)[] points, int start, int end, int depth)
    {
        if (start >= end)
            return null;

        int axis = depth % 3;
        int mid = start + (end - start) / 2;

        Array.Sort(points, start, end - start, Comparer<(Vector3 Point, int Index)>.Create(
            (a, b) => GetAxisValue(a.Point, axis).CompareTo(GetAxisValue(b.Point, axis))));

        var node = new Node
        {
            Point = points[mid].Point,
            Index = points[mid].Index,
            Axis = axis
        };
        node.Left = Build(points, start, mid, depth + 1);
        node.Right = Build(points, mid + 1, end, depth + 1);
        return node;
    }

    private static float GetAxisValue(Vector3 v, int axis) => axis switch
    {
        0 => v.X,
        1 => v.Y,
        _ => v.Z
    };

    // Index into the array the tree was built from, or -1 if the tree is empty.
    public (int index, float distanceSquared) FindNearest(Vector3 query)
    {
        int bestIndex = -1;
        float bestDistSq = float.MaxValue;
        SearchNearest(_root, query, ref bestIndex, ref bestDistSq);
        return (bestIndex, bestDistSq);
    }

    private static void SearchNearest(Node? node, Vector3 query, ref int bestIndex, ref float bestDistSq)
    {
        if (node == null)
            return;

        float distSq = Vector3.DistanceSquared(query, node.Point);
        if (distSq < bestDistSq)
        {
            bestDistSq = distSq;
            bestIndex = node.Index;
        }

        float axisValue = GetAxisValue(query, node.Axis);
        float nodeAxisValue = GetAxisValue(node.Point, node.Axis);
        float diff = axisValue - nodeAxisValue;

        Node? near = diff <= 0 ? node.Left : node.Right;
        Node? far = diff <= 0 ? node.Right : node.Left;

        SearchNearest(near, query, ref bestIndex, ref bestDistSq);

        if (diff * diff < bestDistSq)
        {
            SearchNearest(far, query, ref bestIndex, ref bestDistSq);
        }
    }
}
