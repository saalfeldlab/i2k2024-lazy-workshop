package org.janelia.saalfeldlab.i2k2024.util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import net.imglib2.Interval;
import net.imglib2.util.Intervals;

public interface Grid {

	/**
	 * Create a {@link List} of grid blocks that, for each grid cell, contains
	 * the world coordinate offset, the size of the grid block, and the
	 * grid-coordinate offset.  The spacing for input grid and output grid
	 * are independent, i.e. world coordinate offsets and cropped block-sizes
	 * depend on the input grid, and the grid coordinates of the block are
	 * specified on an independent output grid.  It is assumed that
	 * gridBlockSize is an integer multiple of outBlockSize.
	 *
	 * @param dimensions
	 * @param gridBlockSize
	 * @param outBlockSize
	 * @return
	 */
	public static List<long[][]> create(
			final long[] dimensions,
			final int[] gridBlockSize,
			final int[] outBlockSize) {

		final var n = dimensions.length;
		final var gridBlocks = new ArrayList<long[][]>();

		final var offset = new long[n];
		final var gridPosition = new long[n];
		final var longCroppedGridBlockSize = new long[n];
		for (int d = 0; d < n;) {
			cropBlockDimensions(dimensions, offset, outBlockSize, gridBlockSize, longCroppedGridBlockSize, gridPosition);
				gridBlocks.add(
						new long[][]{
							offset.clone(),
							longCroppedGridBlockSize.clone(),
							gridPosition.clone()
						});

			for (d = 0; d < n; ++d) {
				offset[d] += gridBlockSize[d];
				if (offset[d] < dimensions[d])
					break;
				else
					offset[d] = 0;
			}
		}
		return gridBlocks;
	}

	/**
	 * Create a {@link List} of grid blocks that, for each grid cell, contains
	 * the world coordinate offset, the size of the grid block, and the
	 * grid-coordinate offset.
	 *
	 * @param dimensions
	 * @param blockSize
	 * @return
	 */
	public static List<long[][]> create(
			final long[] dimensions,
			final int[] blockSize) {

		return create(dimensions, blockSize, blockSize);
	}


	/**
	 * Create a {@link List} of grid block offsets in world coordinates
	 * covering an {@link Interval} at a given spacing.
	 *
	 * @param interval
	 * @param spacing
	 * @return
	 */
	public static List<long[]> createOffsets(
			final Interval interval,
			final int[] spacing) {

		final var n = interval.numDimensions();
		final var offsets = new ArrayList<long[]>();

		final var offset = Intervals.minAsLongArray(interval);
		for (int d = 0; d < n;) {
			offsets.add(offset.clone());

			for (d = 0; d < n; ++d) {
				offset[d] += spacing[d];
				if (offset[d] <= interval.max(d))
					break;
				else
					offset[d] = interval.min(d);
			}
		}
		return offsets;
	}

	/**
	 * Returns the grid coordinates of a given offset for a min coordinate and
	 * a grid spacing.
	 *
	 * @param offset
	 * @param min
	 * @param spacing
	 * @return
	 */
	public static long[] gridCell(
			final long[] offset,
			final long[] min,
			final int[] spacing) {

		final var gridCell = new long[offset.length];
		Arrays.setAll(gridCell, i -> (offset[i] - min[i]) / spacing[i]);
		return gridCell;
	}

	/**
	 * Returns the long coordinates <= scaled double coordinates.
	 *
	 * @param doubles
	 * @param scale
	 * @return
	 */
	private static long[] floorScaled(final double[] doubles, final double scale) {

		final var floorScaled = new long[doubles.length];
		Arrays.setAll(floorScaled, i -> (long)Math.floor(doubles[i] * scale));
		return floorScaled;
	}

	/**
	 * Returns the long coordinate >= scaled double coordinates.
	 *
	 * @param doubles
	 * @param scale
	 * @return
	 */
	private static long[] ceilScaled(final double[] doubles, final double scale) {

		final var ceilScaled = new long[doubles.length];
		Arrays.setAll(ceilScaled, i -> (long)Math.ceil(doubles[i] * scale));
		return ceilScaled;
	}

	/**
	 * Crops the dimensions of a {@link DataBlock} at a given offset to fit
	 * into and {@link Interval} of given dimensions.  Fills long and int
	 * version of cropped block size.  Also calculates the grid raster position
	 * assuming that the offset divisible by block size without remainder.
	 *
	 * @param max
	 * @param offset
	 * @param blockSize
	 * @param croppedBlockSize
	 * @param intCroppedBlockDimensions
	 * @param gridPosition
	 */
	private static void cropBlockDimensions(
			final long[] dimensions,
			final long[] offset,
			final int[] outBlockSize,
			final int[] blockSize,
			final long[] croppedBlockSize,
			final long[] gridPosition) {

		for (int d = 0; d < dimensions.length; ++d) {
			croppedBlockSize[d] = Math.min(blockSize[d], dimensions[d] - offset[d]);
			gridPosition[d] = offset[d] / outBlockSize[d];
		}
	}
}
