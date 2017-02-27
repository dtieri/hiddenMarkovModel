#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <Eigen/Dense>
using namespace Eigen;
using namespace std;

class hmm {
  
public:
  hmm(const char* fileName);

  ArrayXXd matchEmissionProbs;
  ArrayXXd insertEmissionProbs;
  ArrayXXd matchTransProbs;
  ArrayXXd insertTransProbs;
  ArrayXXd deleteTransProbs;
  int numSequences;
  int lengthSequence;
  int numBases;

private:
// hidden data from outside world
  void getData (const char* fileName);
  void getState();
  void getMatchIndices();
  void getBaseCounts();
  void getEmissionProbs();
  void getTransProbs();
  vector<string> sequences;
  ArrayXd state;
  ArrayXd matchInd;
  ArrayXXd baseCounts;
  double pseudoCount;
  int numMatchStates;
};

int main ()
{
  hmm hmm1("trainingSequence.txt");
  //cout<<hmm1.baseCounts.block(0,0,hmm1.numBases,hmm1.lengthSequence)<<'\n'<<'\n';
  //cout<<hmm1.matchEmissionProbs.block(0,0,hmm1.numBases,16)<<'\n'<<'\n';
  //cout<<hmm1.insertEmissionProbs.block(0,0,hmm1.numBases,16)<<'\n'<<'\n';
  //cout<<hmm1.numSequences<<'\n'<<hmm1.lengthSequence<<'\n';
  return 0;
}

// Constructor
hmm::hmm(const char* fileName){
  getData(fileName);
  numSequences = sequences.size();
  lengthSequence = sequences[0].size();
  getState();
  numMatchStates=state.sum();
  getMatchIndices();
  numBases = 4;
  pseudoCount = 1.0;
  getBaseCounts();
  getEmissionProbs();
  getTransProbs();
};

// Import data from file and format
void hmm::getData (const char* fileName) {
  ifstream myfile (fileName);
  string line;
  vector<string> list;
  int linesize=0;
  if (myfile.is_open())
    {
      while ( getline (myfile,line) )
	{
	  line.erase(0,line.find(' '));
	  line.erase(0,line.find_first_not_of(' '));
	  list.push_back(line);
	  if(linesize==0){linesize=line.size();}
	  else if(linesize!=line.size()){cout<<"Unequal Sequence Lengths"<<'\n';}
	  //cout << list.size() << '\n'<< line.size() << '\n';
	}
      myfile.close();
      sequences=list;
    }
  else cout << "Unable to open file\n"; 
}

// Count the bases in each column and determine if its a match or insertion state. 1 match state, 0 insertion state
void hmm::getState(){
  int insertionCounts=0;
  state = ArrayXd::Zero(lengthSequence);
  for(int ii=0;ii<lengthSequence;ii=ii+1)
    {
      for(int jj=0;jj<numSequences;jj=jj+1)
	{
	  if(sequences[jj].at(ii)=='-'){++insertionCounts;}   
	};
      if(numSequences>=2*insertionCounts){state(ii)=1;}
      insertionCounts=0;
    };
}
// Get the match state indices
void hmm::getMatchIndices(){
  matchInd = ArrayXd::Zero(numMatchStates);
  int jj=0;
  for(int ii=0;ii<lengthSequence;ii=ii+1)
    {
      if(state(ii)==1){matchInd(jj)=ii;++jj;}
    };
  cout<<matchInd<<'\n'<<numMatchStates<<'\n'<<matchInd.size()<<'\n';
}

// Count the bases in each column
void hmm::getBaseCounts(){
  baseCounts = ArrayXXd::Zero(numBases, lengthSequence);
  for(int ii=0;ii<lengthSequence;ii=ii+1)
    {
      for(int jj=0;jj<numSequences;jj=jj+1)
	{
	  if(sequences[jj].at(ii)=='A'){++baseCounts(0,ii);}
	  else if(sequences[jj].at(ii)=='C'){++baseCounts(1,ii);}
	  else if(sequences[jj].at(ii)=='G'){++baseCounts(2,ii);}
	  else if(sequences[jj].at(ii)=='T'){++baseCounts(3,ii);}
	  else if(sequences[jj].at(ii)!='-'){cout << "Error, Character other than A,C,G,T,- found\n";}    
	};
    };
  //cout<<baseCounts.block(0,0,numBases,lengthSequence)<<'\n';
}

// Get the emission probabilities
void hmm::getEmissionProbs(){
  matchEmissionProbs = ArrayXXd::Zero(numBases, numMatchStates);
  insertEmissionProbs = ArrayXXd::Zero(numBases, numMatchStates);
  int jj=0;
  for(int ii=0;ii<lengthSequence;ii=ii+1)
    {
      if(state(ii)==1)
	{
	matchEmissionProbs.col(jj)=baseCounts.col(ii);
	jj=jj+1;
	}
      else
	{
	insertEmissionProbs.col(jj-1)=insertEmissionProbs.col(jj-1) + baseCounts.col(ii);
	}
    };
  for(int jj=0;jj<numMatchStates;jj=jj+1)
    {
      matchEmissionProbs.col(jj)=(matchEmissionProbs.col(jj)+pseudoCount)/((matchEmissionProbs.col(jj)+pseudoCount).sum());
      if(insertEmissionProbs.col(jj).sum()>0)
	{
	  insertEmissionProbs.col(jj)=(insertEmissionProbs.col(jj)+pseudoCount)/((insertEmissionProbs.col(jj)+pseudoCount).sum());
	}
    };
 }

// Get the Transition probabilities
void hmm::getTransProbs(){
  matchTransProbs = ArrayXXd::Zero(3, numMatchStates-1);
  insertTransProbs = ArrayXXd::Zero(3, numMatchStates-1);
  deleteTransProbs = ArrayXXd::Zero(3, numMatchStates-1);
  int numInserts;
  for(int ii=0;ii<numMatchStates-1;ii=ii+1) // loop through all match states
    {
      numInserts=matchInd(ii+1)-matchInd(ii)-1;
      for(int jj=0;jj<numSequences;jj=jj+1) // loop through all rows
	{
	  if(isupper(sequences[jj].at(matchInd(ii)))) 
	    {
	      if(isupper(sequences[jj].at(matchInd(ii+1)))) 
		{
		  ++matchTransProbs(0,ii); // m-m transitions
		}
	      else
		{
		  ++matchTransProbs(2,ii); // m-d transitions
		}
	      for(int kk=1;kk<=numInserts;++kk) // Only accesed when numInserts>0
		{
		  if(isupper(sequences[jj].at(matchInd(ii)+kk)))
		    {
		      ++matchTransProbs(1,ii); // m-i transitions
		      break;
		    }
		}
	    }
	  else
	    {
	      if(isupper(sequences[jj].at(matchInd(ii+1))))
		{
		  ++deleteTransProbs(0,ii); // d-m transitions
		}
	      else
		{
		  ++deleteTransProbs(2,ii); // d-d transitions
		}
	      for(int kk=1;kk<=numInserts;++kk) // Only accesed when numInserts>0
		{
		  if(isupper(sequences[jj].at(matchInd(ii)+kk)))
		    {
		      ++deleteTransProbs(1,ii); // d-i transitions 
		      break;
		    }
		}
	    }
	  if(isupper(sequences[jj].at(matchInd(ii+1))))
	    {
	      for(int kk=1;kk<=numInserts;++kk) // Only accesed when numInserts>0
		{
		  if(isupper(sequences[jj].at(matchInd(ii+1)-kk))) // Starts at matchInd(ii+1) and goes in reverse
		    {
		      ++insertTransProbs(0,ii); // i-m transitions 
		      break;
		    }
		}
	    }
	  else
	    {
	      for(int kk=1;kk<=numInserts;++kk) // Only accesed when numInserts>0
		{
		  if(isupper(sequences[jj].at(matchInd(ii+1)-kk))) // Starts at matchInd(ii+1) and goes in reverse
		    {
		      ++insertTransProbs(2,ii); // i-d transitions
		      break;
		    }
		}
	    }
	  for(int kk=1;kk<numInserts;++kk) // Only accesed when numInserts>0
	    {
	      if(isupper(sequences[jj].at(matchInd(ii)+kk)))
		{
		  if(isupper(sequences[jj].at(matchInd(ii)+kk+1)))
		    {
		      ++insertTransProbs(1,ii); // i-i transitions
		    }
		}
	    }
	    
	}
    }
  cout<<matchTransProbs<<'\n'<<'\n';
  cout<<insertTransProbs<<'\n'<<'\n';
  cout<<deleteTransProbs<<'\n'<<'\n';
}
