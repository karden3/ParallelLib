program main

    use mpi
	
	implicit none

	integer :: rank, numprocs, ierr, num_row, num_col
	integer :: comm_cart, rank_cart
	integer :: comm_row, comm_col
	integer :: M, N, M_part, N_part, iter
	integer :: i, j, k, s
	integer :: Steps
	real(8) :: delta2
	real(8) :: nu
	real(8) :: time_start, elapsed_time, total_time
	real(8), allocatable :: A_part(:,:), x_part(:), b_part(:)
	real(8), allocatable :: x(:), x_classic_part(:), x_classic(:)
	real(8), allocatable :: x_model(:)
	real(8), allocatable :: b_part_temp(:)
	integer, allocatable :: rcounts_M(:), rcounts_N(:)
	integer, allocatable :: displs_M(:), displs_N(:) 
	integer :: dims(2), coords(2), my_row, my_col
	logical :: periods(2)
	real(8), parameter :: pi = 3.1415926535897932384626433832795
	real(8) :: alpha
	integer, allocatable :: seed(:)
	integer :: n_seed
   
	call MPI_Init(ierr)
	call MPI_Comm_size(MPI_COMM_WORLD, numprocs, ierr)
	
	N = 20 
	M = 30
	
	Steps = 100
	
	nu = 0.5
	alpha = 0.d0
	
	num_row = int(sqrt(real(numprocs, kind(0d0))))
	num_col = int(sqrt(real(numprocs, kind(0d0))))
	
	dims(1) = num_row; 	dims(2) = num_col
	periods(1) = .TRUE.; periods(2) = .TRUE.

	call MPI_Cart_create(MPI_COMM_WORLD, 2, dims, &
						 periods, .TRUE., comm_cart, ierr)
	
	call MPI_Comm_rank(comm_cart, rank_cart, ierr)	
	
	call MPI_Cart_coords(comm_cart, rank_cart, 2, coords, ierr)
	
	my_row = coords(1)
	my_col = coords(2)
	
	call MPI_Comm_split(comm_cart, mod(rank_cart, num_col), &
						rank_cart, comm_col, ierr)
	call MPI_Comm_split(comm_cart, int(rank_cart/num_col), &
						rank_cart, comm_row, ierr)

	allocate(rcounts_M(0:num_row-1), displs_M(0:num_row-1))
	allocate(rcounts_N(0:num_col-1), displs_N(0:num_col-1))

	call auxiliary_arrays(M, num_row, rcounts_M, displs_M) 
	call auxiliary_arrays(N, num_col, rcounts_N, displs_N) 
	
	M_part = rcounts_M(my_row)
	N_part = rcounts_N(my_col)
	
	allocate(A_part(M_part, N_part), x_part(N_part), &
			 b_part(M_part), b_part_temp(M_part), &
			 x_classic_part(N_part))
	
	call random_seed(size=n_seed)
	allocate(seed(n_seed))
	seed = rank_cart
	call random_seed(put=seed)
	call random_number(A_part)
		
	if (rank_cart == 0) then
		allocate(x_model(N))
		do i = 1,N
			x_model(i) = real(sin(2*pi*(i-1)/(N-1)), kind(0d0))
		end do
	end if
	
	if (rank_cart < num_col) then
		call MPI_Scatterv(x_model,rcounts_N,displs_N,MPI_REAL8,&
						  x_part, N_part, MPI_REAL8, &
						  0, comm_row, ierr)
	endif
	call MPI_Bcast(x_part, N_part, MPI_REAL8, &
				   0, comm_col, ierr)
	b_part_temp = matmul(A_part, x_part)
	call MPI_Allreduce(b_part_temp, b_part, &
					   M_part, MPI_REAL8, &
					   MPI_SUM, comm_row, ierr)
			
	x_part = 0.d0
	delta2 = 0.d0
	
	call MPI_Barrier(comm_cart, ierr)
	time_start = MPI_Wtime()
	
	call Brakhage_parallel_mpi_3(A_part,b_part,x_part,delta2, &
					 M_part, N_part, &
					 comm_row, comm_col, N, nu, Steps)
	
	elapsed_time = MPI_Wtime() - time_start
	call MPI_Reduce(elapsed_time, total_time, 1, MPI_REAL8, &
				    MPI_MAX, 0, comm_cart, ierr)

	if (rank_cart == 0) then
		print '(a6, f9.3)','Time:', total_time
	endif
	
	if (rank_cart == 0) then
		allocate(x(N))
	endif
		
	if (rank_cart < num_col) then
		call MPI_Gatherv(x_part, N_part, MPI_REAL8, &
						 x, rcounts_N, displs_N, MPI_REAL8, &
						 0, comm_row, ierr)
	endif
	
	if (rank_cart == 0) then
		do i=1,N
			print '(f32.28, f32.28)', &
				  x(i), x_model(i)
			
		enddo
	endif
	
	call MPI_Finalize(ierr)

end program

subroutine Brakhage_parallel_mpi_3(A_part,b_part,x_part,delta2, &
					 M_part, N_part, &
					 comm_row, comm_col, N, nu, Steps)

	use mpi
	
	implicit none						 
							 		 
	integer  :: M_part, N_part, N, s, i, j, Steps
	real(8) :: A_part(M_part,N_part),x_part(N_part),b_part(M_part),bb_part(M_part)
	real(8) :: normAsq_part1, normAsq_part2, normAsq, normA
	real(8) :: delta2
	real(8) :: nu
	real(8) :: w
	real(8) :: mu
	real(8) :: c_part(N_part), x_prev_part(N_part)
	real(8) :: Ax_part(M_part), Ax_part_temp(M_part)
	real(8) :: gr_part(N_part), gr_part_temp(N_part)
	real(8) :: criterion, criterion_temp
	integer  :: comm_row, comm_col
	integer  :: status(MPI_STATUS_SIZE)
	integer  :: ierr, info
	
   	call MPI_Info_create(info, ierr)
	call MPI_Info_set(info, "key", "value", ierr)
	
	s = 1
	normAsq_part1 = 0
	
	do j = 1, M_part
		do i = 1, N_part
			normAsq_part1 = normAsq_part1 + (A_part(j,i))**2
		end do
	end do
	call MPI_Allreduce(normAsq_part1, normAsq_part2, &
						1, MPI_REAL8, &
						MPI_SUM, comm_row, ierr)
	call MPI_Allreduce(normAsq_part2, normAsq, &
						1, MPI_REAL8, &
						MPI_SUM, comm_col, ierr)					
	normA = dsqrt(normAsq) 
	
	A_part = A_part / normA
	b_part = b_part / normA
	delta2 = delta2 / (normA**2)
	
	do while (.TRUE.)
		x_prev_part(i) = x_part(i)

		Az_part_temp = matmul(A_part, z_part)

		call MPI_Allreduce(Az_part_temp, Az_part, &
						   M_part, MPI_REAL8, &
						   MPI_SUM, comm_row, ierr)

		b_part = b_part - Az_part
		x_partz_temp = matmul(transpose(A_part), b_part)
		if (s == 1) then
			w = (4*nu + 2)/(4*nu + 1)
			x_prev_part = x_part
			Ax_part_temp = matmul(A_part, x_part)
			call MPI_Allreduce(Ax_part_temp, Ax_part, &
							   M_part, MPI_REAL8, &
							   MPI_SUM, comm_row, ierr)
			bb_part = b_part - Ax_part
			gr_part_temp = matmul(transpose(A_part), bb_part)
			call MPI_Allreduce(gr_part_temp, gr_part, &
							   N_part, MPI_REAL8, &
							   MPI_SUM, comm_col, ierr)
			x_part = x_part + w*gr_part
		else 
			mu = (s - 1)*(2*s - 3)*(2*s + 2*nu - 1)/((s + 2*nu - 1)*(2*s + 4*nu - 1)*(2*s + 2*nu - 3))
			w = 4*((2*s + 2*nu - 1)*(s + nu - 1))/((s + 2*nu - 1)*(2*s + 4*nu - 1))	
			c_part = x_prev_part
			x_prev_part = x_part
            		Ax_part_temp = matmul(A_part, x_part)
            		call MPI_Allreduce(Ax_part_temp, Ax_part, &
							   M_part, MPI_REAL8, &
							   MPI_SUM, comm_row, ierr)
			bb_part = b_part - Ax_part
            		gr_part_temp = matmul(transpose(A_part), bb_part)
			call MPI_Allreduce(gr_part_temp, gr_part, &
							N_part, MPI_REAL8, &
							MPI_SUM, comm_col, ierr)
			x_part = x_part + mu*(x_part - c_part) + w*gr_part
		end if
		    
		s = s + 1		
		Ax_part_temp = matmul(A_part, x_part)
		call MPI_Allreduce(Ax_part_temp, Ax_part, &
							   M_part, MPI_REAL8, &
							   MPI_SUM, comm_row, ierr)
		
		criterion_temp = sum((Ax_part - b_part)**2)
		call MPI_Allreduce(criterion_temp, criterion, &
							1, MPI_REAL8, &
							MPI_SUM, comm_col, ierr)
		if (s >= Steps) then
			call MPI_Info_free(info, ierr)
	    	return
		end if

	end do
	
end subroutine


subroutine auxiliary_arrays(M, num, rcounts, displs) 

	implicit none
    
	integer M, num
	integer ave, res, k
	integer rcounts(0:num-1), displs(0:num-1)
	
	ave = int(M/num)
	res = mod(M, num)
	
	do k = 0,num-1
		if (k < res) then
			rcounts(k) = ave + 1
		else
			rcounts(k) = ave
		endif
		if (k == 0) then
			displs(k) = 0
		else
			displs(k) = displs(k-1) + rcounts(k-1)
		endif
	end do
	
end subroutine
