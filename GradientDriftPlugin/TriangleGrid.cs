using System;
using System.Collections.Generic;
using System.Numerics;
using Serilog;

namespace GradientDriftPlugin;

// Nearest triangle a ray hit: distance along the ray and that triangle's surface.
public readonly struct RaycastHit
{
    public readonly float Distance;
    public readonly SurfaceType Surface;

    public RaycastHit(float distance, SurfaceType surface)
    {
        Distance = distance;
        Surface = surface;
    }
}

// Uniform 3D grids over the track's triangles for ray casts: Amanatides-Woo traversal visits
// only the cells a ray crosses, Moller-Trumbore tests the triangles in each. Built once at
// startup. Ordinary triangles go in a fine grid (FineCellSizeM); the few large ones (ground
// planes, long walls, scenery) in a coarse grid of their own, so they do not fill thousands of
// fine cells. A ray takes the nearer of the two grids' hits: the same hit as testing every
// triangle (HISTORY.md, "Sensor cost").
public class TriangleGrid
{
    private const float FineCellSizeM = 2f;
    private const float LargeTriangleM = 16f;

    private readonly Layer _fine;
    private readonly Layer? _coarse;

    // targetTrianglesPerCell sets the coarse grid's cell size. Triangles spanning more than
    // maxCellSpan cells of their grid are left out so they cannot blow up memory.
    public TriangleGrid(TrackTriangle[] triangles, int targetTrianglesPerCell = 8, int maxCellSpan = 64)
    {
        Vector3 min = new Vector3(float.MaxValue);
        Vector3 max = new Vector3(float.MinValue);
        var small = new List<int>(triangles.Length);
        var large = new List<int>();
        for (int idx = 0; idx < triangles.Length; idx++)
        {
            var t = triangles[idx];
            Vector3 tMin = Vector3.Min(t.V1, Vector3.Min(t.V2, t.V3));
            Vector3 tMax = Vector3.Max(t.V1, Vector3.Max(t.V2, t.V3));
            min = Vector3.Min(min, tMin);
            max = Vector3.Max(max, tMax);
            Vector3 size = tMax - tMin;
            (MathF.Max(size.X, MathF.Max(size.Y, size.Z)) > LargeTriangleM ? large : small).Add(idx);
        }
        if (triangles.Length == 0)
        {
            min = max = Vector3.Zero;
        }
        min -= new Vector3(0.01f);
        max += new Vector3(0.01f);

        _fine = new Layer(triangles, small, min, max, FineCellSizeM, maxCellSpan);
        if (large.Count > 0)
        {
            Vector3 extent = max - min;
            float coarseCell = MathF.Max(MathF.Pow(extent.X * extent.Y * extent.Z * targetTrianglesPerCell / large.Count, 1f / 3f), 0.5f);
            _coarse = new Layer(triangles, large, min, max, coarseCell, maxCellSpan);
        }

        VerboseLog.Information("GradientDrift: TriangleGrid built - {Small} triangles in {FineCells} cells of {FineSize:F1}m, {Large} large ones in {CoarseCells} cells of {CoarseSize:F1}m",
            small.Count, _fine.CellCount, FineCellSizeM, large.Count, _coarse?.CellCount ?? 0, _coarse?.CellSize ?? 0f);
    }

    // First hit within maxDistance along direction (a unit vector), if any. wallsOnly: walls are the
    // only triangles a ray can hit (the contact pins: a rise in the road just ahead must not hide a
    // low wall behind it).
    public bool Raycast(Vector3 origin, Vector3 direction, float maxDistance, out RaycastHit hit, bool wallsOnly = false)
    {
        bool found = _fine.Raycast(origin, direction, maxDistance, out hit, wallsOnly);
        if (_coarse != null
            && _coarse.Raycast(origin, direction, found ? hit.Distance : maxDistance, out RaycastHit coarseHit, wallsOnly)
            && (!found || coarseHit.Distance < hit.Distance))
        {
            hit = coarseHit;
            found = true;
        }
        return found;
    }

    private sealed class Layer
    {
        private readonly TrackTriangle[] _triangles;
        private readonly Dictionary<long, (int Start, int Count)> _cells = new();
        private readonly int[] _indices;
        private readonly Vector3 _min;
        private readonly float _cellSize;
        private readonly int _dimX, _dimY, _dimZ;

        public int CellCount => _cells.Count;
        public float CellSize => _cellSize;

        public Layer(TrackTriangle[] triangles, List<int> members, Vector3 min, Vector3 max, float cellSize, int maxCellSpan)
        {
            _triangles = triangles;
            _min = min;
            _cellSize = cellSize;
            Vector3 extent = max - min;
            _dimX = Math.Max(1, (int)(extent.X / _cellSize) + 1);
            _dimY = Math.Max(1, (int)(extent.Y / _cellSize) + 1);
            _dimZ = Math.Max(1, (int)(extent.Z / _cellSize) + 1);

            var lists = new Dictionary<long, List<int>>();
            int skipped = 0;
            int entries = 0;
            foreach (int idx in members)
            {
                var t = triangles[idx];
                var (x0, y0, z0) = CellIndex(Vector3.Min(t.V1, Vector3.Min(t.V2, t.V3)));
                var (x1, y1, z1) = CellIndex(Vector3.Max(t.V1, Vector3.Max(t.V2, t.V3)));
                if (Math.Max(x1 - x0, Math.Max(y1 - y0, z1 - z0)) + 1 > maxCellSpan)
                {
                    skipped++;
                    continue;
                }

                for (int ix = x0; ix <= x1; ix++)
                for (int iy = y0; iy <= y1; iy++)
                for (int iz = z0; iz <= z1; iz++)
                {
                    long key = Key(ix, iy, iz);
                    if (!lists.TryGetValue(key, out var list))
                    {
                        list = new List<int>();
                        lists[key] = list;
                    }
                    list.Add(idx);
                    entries++;
                }
            }

            _indices = new int[entries];
            int start = 0;
            foreach (var (key, list) in lists)
            {
                list.CopyTo(_indices, start);
                _cells[key] = (start, list.Count);
                start += list.Count;
            }

            if (skipped > 0)
            {
                Log.Warning("GradientDrift: TriangleGrid excluded {Skipped} oversized triangles (AABB spans >{MaxSpan} cells) - likely distant scenery, not track geometry",
                    skipped, maxCellSpan);
            }
        }

        private long Key(int ix, int iy, int iz) => ((long)ix * _dimY + iy) * _dimZ + iz;

        private (int, int, int) CellIndex(Vector3 p)
        {
            int ix = Math.Clamp((int)((p.X - _min.X) / _cellSize), 0, _dimX - 1);
            int iy = Math.Clamp((int)((p.Y - _min.Y) / _cellSize), 0, _dimY - 1);
            int iz = Math.Clamp((int)((p.Z - _min.Z) / _cellSize), 0, _dimZ - 1);
            return (ix, iy, iz);
        }

        public bool Raycast(Vector3 origin, Vector3 direction, float maxDistance, out RaycastHit hit, bool wallsOnly)
        {
            var (cellX, cellY, cellZ) = CellIndex(origin);

            SetupAxis(direction.X, origin.X, _min.X, cellX, out int stepX, out float tMaxX, out float tDeltaX);
            SetupAxis(direction.Y, origin.Y, _min.Y, cellY, out int stepY, out float tMaxY, out float tDeltaY);
            SetupAxis(direction.Z, origin.Z, _min.Z, cellZ, out int stepZ, out float tMaxZ, out float tDeltaZ);

            float bestT = float.PositiveInfinity;
            SurfaceType bestSurface = SurfaceType.None;
            bool found = false;

            for (int guard = 0; guard < 100000; guard++)
            {
                if (_cells.TryGetValue(Key(cellX, cellY, cellZ), out var cell))
                {
                    for (int k = cell.Start; k < cell.Start + cell.Count; k++)
                    {
                        ref readonly var tri = ref _triangles[_indices[k]];
                        if (wallsOnly && tri.Surface != SurfaceType.Wall)
                        {
                            continue;
                        }
                        if (RayIntersectsTriangle(origin, direction, tri.V1, tri.V2, tri.V3, out float t)
                            && t <= maxDistance && t < bestT)
                        {
                            bestT = t;
                            bestSurface = tri.Surface;
                            found = true;
                        }
                    }
                }

                float nextBoundary = MathF.Min(tMaxX, MathF.Min(tMaxY, tMaxZ));

                // A hit is final once it lies inside the cells visited so far.
                if (found && bestT <= nextBoundary)
                {
                    break;
                }
                if (nextBoundary > maxDistance)
                {
                    break;
                }

                if (tMaxX < tMaxY && tMaxX < tMaxZ)
                {
                    cellX += stepX;
                    if (cellX < 0 || cellX >= _dimX) break;
                    tMaxX += tDeltaX;
                }
                else if (tMaxY < tMaxZ)
                {
                    cellY += stepY;
                    if (cellY < 0 || cellY >= _dimY) break;
                    tMaxY += tDeltaY;
                }
                else
                {
                    cellZ += stepZ;
                    if (cellZ < 0 || cellZ >= _dimZ) break;
                    tMaxZ += tDeltaZ;
                }
            }

            hit = found ? new RaycastHit(bestT, bestSurface) : default;
            return found;
        }

        // Per-axis traversal state: which way to step, the ray distance to the next cell boundary,
        // and the ray distance between boundaries.
        private void SetupAxis(float dir, float origin, float gridMin, int cellIndex, out int step, out float tMax, out float tDelta)
        {
            const float epsilon = 1e-12f;
            if (dir > epsilon)
            {
                step = 1;
                float boundary = gridMin + (cellIndex + 1) * _cellSize;
                tMax = (boundary - origin) / dir;
                tDelta = _cellSize / dir;
            }
            else if (dir < -epsilon)
            {
                step = -1;
                float boundary = gridMin + cellIndex * _cellSize;
                tMax = (boundary - origin) / dir;
                tDelta = _cellSize / -dir;
            }
            else
            {
                step = 0;
                tMax = float.PositiveInfinity;
                tDelta = float.PositiveInfinity;
            }
        }
    }

    // Moller-Trumbore (1997).
    private static bool RayIntersectsTriangle(Vector3 origin, Vector3 direction, Vector3 v1, Vector3 v2, Vector3 v3, out float t)
    {
        const float epsilon = 1e-6f;
        t = 0f;
        Vector3 edge1 = v2 - v1;
        Vector3 edge2 = v3 - v1;
        Vector3 h = Vector3.Cross(direction, edge2);
        float a = Vector3.Dot(edge1, h);
        if (a > -epsilon && a < epsilon) return false;
        float f = 1f / a;
        Vector3 s = origin - v1;
        float u = f * Vector3.Dot(s, h);
        if (u < 0f || u > 1f) return false;
        Vector3 q = Vector3.Cross(s, edge1);
        float v = f * Vector3.Dot(direction, q);
        if (v < 0f || u + v > 1f) return false;
        t = f * Vector3.Dot(edge2, q);
        return t > epsilon;
    }
}
