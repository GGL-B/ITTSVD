import numpy as np
from scipy.sparse import csr_matrix


def hungarian(A):
    """
    HUNGARIAN Solve the Assignment problem using the Hungarian method
    parameter:
    A - a square cost matrix.
    C - the optimal assignment
    T - the cost of the optimal assignment
    """

    m, n = A.shape
    if m != n:
        raise ValueError("Cost matrix must be square")
    orig = A.copy()

    # ==== Reduce matrix. ====
    A = hminired(A)

    # ==== initial assignment. ====
    A, C, U = hminiass(A)

    # ==== Repeat while we have unassigned rows. ====
    while U[n] != 0:  # Start with no path, no unchecked zeros, and no unexplored rows.
        LR = np.zeros(n, dtype=int)
        LC = np.zeros(n, dtype=int)
        CH = np.zeros(n, dtype=int)
        RH = np.zeros(n + 1, dtype=int)
        RH[-1] = -1

        #No labelled columns.
        SLC = []
        SLR = []

        # Start path in first unassigned row.
        r = int(U[n])
        LR[r-1]=-1
        #Insert row first in labelled row set
        SLR.append(r)
        # Repeat until we manage to find an assignable zero.
        while True:
            #If there are free zeros in row r
            if A[r - 1, n] != 0:
                l = int(-A[r - 1, n])  # get column of first free zero
                # Add boundary check
                if r - 1 < A.shape[0] and l - 1 < A.shape[1] and A[r - 1, l - 1] != 0 and RH[r - 1] == 0:
                    # Insert row r first in unexplored list
                    RH[r - 1] = RH[n]
                    RH[n] = r
                    #  Mark in which column the next unexplored zero in this row
                    CH[r - 1] = -A[r - 1, l - 1]
            else:
                # If all rows are explored
                if RH[n] <= 0:
                    #Reduce matrix.
                    A, CH, RH = hmreduce(A, CH, RH, LC, LR, SLC, SLR)
                r = int(RH[n])
                if r < 0:
                    break
                #Get column of next free zero in row r
                l = int(CH[r - 1])
                # l = CH[r - 1]
                #  Advance "column of next free zero"
                CH[r - 1] = -A[r - 1, l - 1]
                # If this zero is last in the list
                if A[r - 1, l - 1] == 0:
                    # remove row r from unexplored list
                    RH[n] = RH[r - 1]
                    RH[r - 1] = 0

            # while the column l is labelled in path
            while LC[l - 1] != 0:
                # If row r is explored
                if l - 1 >= len(LC):
                    break
                #If all rows are explored
                if RH[r - 1] == 0:
                    if RH[n] <= 0:
                        #Reduce cost matrix.
                        A, CH, RH = hmreduce(A, CH, RH, LC, LR, SLC, SLR)
                    r = int(RH[n])
                    if r < 0:
                        break
                #Get column of next free zero in row r
                l = int(CH[r - 1])
                # Advance "column of next free zero"
                CH[r - 1] = -A[r - 1, l - 1]
                if A[r - 1, l - 1] == 0:
                    RH[n] = RH[r - 1]
                    RH[r - 1] = 0

            if C[l - 1] == 0:
                #Flip all zeros along the path in LR,LC.
                A, C, U = hmflip(A, C, LC, LR, U, l, r)
                C = C.astype(int)
                #and exit to continue with next unassigned row.
                break
            else:
                # Label column l with row r
                LC[l - 1] = r
                # Add l to the set of labelled columns.
                SLC.append(l)
                # Continue with the row assigned to column l
                r = int(C[l - 1])  # 确保是整数
                #  Label row r with column l
                LR[r - 1] = l
                # Add r to the set of labelled rows.
                SLR.append(r)

    # ==== Calculate the total cost ====
    row_indices = C - 1
    col_indices = np.arange(n)
    mask = np.zeros_like(orig, dtype=bool)
    mask[row_indices, col_indices] = True
    T = np.sum(orig[mask])

    return C, T


def hminired(A):
    """
    HMINIRED Initial reduction of cost matrix for the Hungarian method.
    param A - the unreduced cost matris
    B - the reduced cost matrix with linked zeros in each row.
    """
    A = np.array(A, dtype=float)
    m, n = A.shape
    #  Subtract column-minimum values from each column.
    colMin = np.min(A, axis=0)
    A = A - np.tile(colMin, (n, 1))
    # Subtract row-minimum values from each row.
    rowMin = np.min(A, axis=1)
    A = A - rowMin.reshape(-1, 1)

    # Get positions of all zeros.
    i, j = np.where(np.abs(A) < 1e-10)
    # Extend A to give room for row zero list header column.
    A = np.hstack([A, np.zeros((m, 1))])

    for k in range(n):
        # Get all column in this row.
        mask = (i == k)
        cols = j[mask]
        if len(cols) > 0:
            A[k, n] = -cols[0] - 1
            for idx in range(len(cols)):
                col = cols[idx]
                if idx + 1 < len(cols):
                    next_col = cols[idx + 1]
                    A[k, col] = -next_col - 1
                else:
                    A[k, col] = 0
        else:
            A[k, n] = 0

    return A


#HMINIASS Initial assignment of the Hungarian method.
def hminiass(A):
    n, np1 = A.shape
    if np1 != n + 1:
        raise ValueError("The input matrix A must have dimensions of n × (n+1)")
    #Initalize return vectors.
    C = np.zeros(n, dtype=int)
    U = np.zeros(n + 1, dtype=int)

    #Initialize last/next zero "pointers".
    LZ = np.zeros(n, dtype=int)
    NZ = np.zeros(n, dtype=int)

    for i in range(n):
        # Set j to first unassigned zero in row i.
        lj = n

        temp_j = int(-A[i, lj]) if A[i, lj] < 0 else 0

        #  Repeat until we have no more zeros (j==0) or we find a zero in an unassigned column (c(j)==0).
        while temp_j != 0 and C[temp_j - 1] != 0:
            #Advance lj and j in zero list.
            lj = temp_j - 1
            temp_j = int(-A[i, lj]) if A[i, lj] < 0 else 0

        if temp_j != 0:
            # We found a zero in an unassigned column.
            j = temp_j - 1
            C[j] = i + 1

            # Remove A(i,j) from unassigned zero list.
            A[i, lj] = A[i, j]
            NZ[i] = int(-A[i, j]) if A[i, j] < 0 else 0
            LZ[i] = lj
            #Indicate A(i,j) is an assigned zero.
            A[i, j] = 0
        else:
            #  Check all zeros in this row.
            lj = n
            temp_j = int(-A[i, lj]) if A[i, lj] < 0 else 0

            #Check all zeros in this row for a suitable zero in another row.
            while temp_j != 0:
                j = temp_j - 1
                temp_r = C[j]
                # Stop if we find an unassigned column.
                if temp_r == 0:
                    C[j] = i + 1
                    A[i, lj] = A[i, j]
                    NZ[i] = int(-A[i, j]) if A[i, j] < 0 else 0
                    LZ[i] = lj
                    A[i, j] = 0
                    break

                r = temp_r - 1
                #Advance one step in list.
                lm = LZ[r]
                temp_m = int(NZ[r])

                #  Check all unchecked zeros in free list of this row.
                while temp_m != 0:
                    m = temp_m - 1
                    if C[m] == 0:
                        break
                    lm = m
                    temp_m = int(-A[r, lm]) if A[r, lm] < 0 else 0

                if temp_m == 0:
                    #  We failed on row r. Continue with next zero on row i.
                    lj = j
                    j = int(-A[i, lj]) if A[i, lj] < 0 else 0
                else:
                    # We found a zero in an unassigned column.
                    m = temp_m - 1
                    #  Replace zero at (r,m) in unassigned list with zero at (r,j)
                    A[r, lm] = -(j + 1)
                    A[r, j] = A[r, m]
                    # Update last/next pointers in row r.
                    NZ[r] = int(-A[r, m]) if A[r, m] < 0 else 0
                    LZ[r] = j
                    #Mark A(r,m) as an assigned zero in the matrix
                    A[r, m] = 0
                    C[m] = r + 1

                    # Remove A(i,j) from unassigned list.
                    A[i, lj] = A[i, j]
                    NZ[i] = int(-A[i, j]) if A[i, j] < 0 else 0
                    LZ[i] = lj
                    A[i, j] = 0
                    C[j] = i + 1
                    # Stop search.
                    break

    #Create vector with list of unassigned rows.
    assigned = np.zeros(n, dtype=int)
    assigned_rows = C[C != 0] - 1
    if len(assigned_rows) > 0:
        assigned[assigned_rows] = 1
    empty_rows = np.where(assigned == 0)[0] + 1

    #Create vector with linked list of unassigned rows.
    U = np.zeros(n + 1, dtype=int)
    if len(empty_rows) > 0:
        U[n] = empty_rows[0]
        for k in range(len(empty_rows) - 1):
            U[empty_rows[k]] = empty_rows[k + 1]
        U[empty_rows[-1]] = 0
    else:
        U[n] = 0

    return A, C, U


def hmflip(A, C, LC, LR, U, l, r):
    """
    HMFLIP Flip assignment state of all zeros along a path.
    Input:
          A   - the cost matrix.成本矩阵。
          C   - the assignment vector.分配向量。
          LC  - the column label vector.列标签向量。
          LR  - the row label vector.行标签向量
          U   - the
          r,l - position of last zero in path.路径中最后一个零的位置。
    Output:
         A   - updated cost matrix.更新成本矩阵
         C   - updated assignment vector.更新的分配向量。
         U   - updated unassigned row list vector.更新了未分配的行列表矢量。
    """

    l = l - 1
    r = r - 1
    n = A.shape[0]

    while True:
        # Move assignment in column l to row r.
        C[l] = r + 1

        # Find the column with a value of - l in row r
        mask = np.abs(A[r, :] + (l + 1)) < 1e-10
        m = np.where(mask)[0]

        if m.size > 0:
            m = m[0]
            # Link past this zero.
            A[r, m] = A[r, l]

        A[r, l] = 0

        # If this was the first zero of the path.
        if LR[r] < 0:
            # remove row from unassigned row list and return
            U[n] = U[r]
            U[r] = 0
            return A, C, U
        else:
            #Move back in this row along the path and get column of next zero
            l = LR[r] - 1


            A[r, l] = A[r, n]
            A[r, n] = -(l + 1)

            # Continue back along the column to get row of next zero in path.
            r = LC[l] - 1
            return A, C, U


def hmreduce(A, CH, RH, LC, LR, SLC, SLR):
    """
    HMREDUCE Reduce parts of cost matrix in the Hungerian method.
    Input:
         A   - Cost matrix.成本矩阵。
        CH  - vector of column of 'next zeros' in each row.每行中“下一个零”列的向量。
        RH  - vector with list of unexplored rows.带有未探索行列表的矢量。
        LC  - column labels.列标签。
        RC  - row labels.行标签。
        SLC - set of column labels.一组列标签。
        SLR - set of row labels.行标签集。

   Output:
        A   - Reduced cost matrix.降低成本矩阵。
        CH  - Updated vector of 'next zeros' in each row.更新了每行中“下一个零”的矢量
        RH  - Updated vector of unexplored rows
    """
    n = A.shape[0]

    # ==== Processing uncovered elements ====
    # Find which rows are covered
    covered_rows = (LR == 0)
    covered_cols = (LC != 0)

    #  Find which columns are covered
    r_uncovered = np.where(~covered_rows)[0]
    c_uncovered = np.where(~covered_cols)[0]

    if len(r_uncovered) == 0 or len(c_uncovered) == 0:
        return A, CH, RH

    # Get minimum of uncovered elements
    m = np.min(A[np.ix_(r_uncovered, c_uncovered)])
    # Subtract minimum from all uncovered elements.
    A[np.ix_(r_uncovered, c_uncovered)] -= m

    # ==== Check all uncovered columns ====
    for j in c_uncovered:
        for temp_i in SLR:
            i = temp_i - 1
            if np.abs(A[i, j]) < 1e-10:
                # If the row is not in unexplored list
                if RH[i] == 0:
                    # insert it first in unexplored list
                    RH[i] = RH[n]
                    RH[n] = i + 1
                    CH[i] = j + 1

                # Find last unassigned zero on row i
                row = A[i, :]
                negative_indices = np.where(row < 0)[0]
                cols_in_list = []
                for idx in negative_indices:
                    col_val = int(-row[idx])
                    cols_in_list.append(col_val - 1)

                if len(cols_in_list) == 0:
                    l = n  # No zeros in the list
                else:
                    valid_cols = []
                    for col in cols_in_list:
                        # Ensure that the column index is valid and the element is zero
                        if col < n and np.abs(A[i, col]) < 1e-10:
                            valid_cols.append(col)

                    if len(valid_cols) > 0:
                        l = valid_cols[-1]
                    else:
                        l = n

                # Append this zero to end of list
                A[i, l] = -(j + 1)

    # ==== Handling double coverage elements ====
    covered_row_indices = np.where(covered_rows)[0]
    covered_col_indices = np.where(covered_cols)[0]

    if len(covered_row_indices) > 0 and len(covered_col_indices) > 0:

        for i in covered_row_indices:
            for j in covered_col_indices:
                if A[i, j] <= 0:
                    target_val = -(j + 1)
                    #Find zero before this in this row.
                    lj_indices = np.where(np.abs(A[i, :] - target_val) < 1e-10)[0]
                    if len(lj_indices) > 0:
                        lj = lj_indices[0]
                        # Mark it as assigned
                        A[i, lj] = A[i, j]
                        A[i, j] = 0

        # Add the minimum value back to the double coverage area
        A[np.ix_(covered_row_indices, covered_col_indices)] += m

    return A, CH, RH


