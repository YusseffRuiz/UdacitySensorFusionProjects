/* \author Aaron Brown */
// Quiz on implementing kd tree




// Structure to represent node of kd tree
struct Node
{
	std::vector<float> point;
	int id;
	Node* left;
	Node* right;

	Node(std::vector<float> arr, int setId)
	:	point(arr), id(setId), left(NULL), right(NULL)
	{}
};

struct KdTree
{
	Node* root;

	KdTree()
	: root(NULL)
	{}


	void insertHelper(Node** node, uint depth, const std::vector<float>& point, int id)
	{
		if(*node == NULL)
		{
			*node = new Node(point, id);
		}
		else 
		{
			uint in = depth % 3; // even = movement in Y, odd = movement in X

			if(point[in] < ((*node) -> point[in])){
				insertHelper(&((*node)->left), depth + 1, point, id);
			}
			else{
				insertHelper(&((*node)->right), depth + 1, point, id);
			}
		}
	}

	void insert(const std::vector<float>& point, int id)
	{
		// TODO: Fill in this function to insert a new point into the tree
		// the function should create a new node and place correctly with in the root 
		insertHelper(&root, 0, point, id);



	}


	void searchHelper(const std::vector<float>& target, Node* node, int depth, float distanceTol, std::vector<int>& ids)
	{
		int in = depth % 3; // 2 = movement in Z, 1 = movement in Y, 0 = movement in X
		//float insideBox = 2*distanceTol;
		
		if(node != NULL){

			float dx = fabs(node->point[0] - target[0]);
			float dy = fabs(node->point[1] - target[1]);

			if(dx <= distanceTol && dy <= distanceTol) {
				float distance = std::hypotf(dx, dy);
				if(distance <= distanceTol){
					ids.push_back((node->id));
				}
			}

			if((target[in] - distanceTol) < node->point[in]){
				searchHelper(target, node-> left, depth + 1, distanceTol, ids);
		
	
			}
			if(target[in] + distanceTol > node->point[in]){
				searchHelper(target, node->right, depth+1, distanceTol, ids);
			}
				
		}
	}

	// return a list of point ids in the tree that are within distance of target
	std::vector<int> search(const std::vector<float>& target, float distanceTol)
	{
		std::vector<int> ids;
		searchHelper(target, root, 0, distanceTol, ids);
		return ids;
	}
	

};




