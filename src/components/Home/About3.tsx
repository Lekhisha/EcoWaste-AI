
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faComments } from "@fortawesome/free-regular-svg-icons";
import icon1 from "../../assests/icon1.png";
import icon2 from "../../assests/icon2.png";
import bg3 from "../../assests/bg3.png";

const bgstyle2 = {
  backgroundImage: `url(${bg3})`,
  height: "100%",
  width: "100%",
  backgroundPosition: "center",
  backgroundRepeat: "no-repeat",
  backgroundSize: "cover",
};

const About3 = () => {
  return (
    <>
      <div className="relative py-14 " style={bgstyle2}>
        <div className="container bg-white/20 backdrop-blur-md rounded-2xl shadow-lg py-4">
          <div className="text-cyan-950 text-4xl font-semibold text-center">
            Chatbot Assistance
          </div>
          <div className="pb-10">
            <p className="text-cyan-950 font-semibold pt-4 w-1/2 mx-auto text-center">
              Our intelligent AI chatbot is your personal waste management
              assistant. Ask it anything about waste disposal, recycling rules,
              or eco-friendly practices, and get reliable, instant support. It's
              like having an expert by your side 24/7.
            </p>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="text-center ">
             <FontAwesomeIcon icon={faComments} className="text-5xl pb-4 text-green-700"/>

              <div className="text-cyan-950 text-2xl font-semibold pb-4">
                Instant Answers
              </div>
              <p className="text-cyan-950 font-semibold">
                Get immediate responses to your waste disposal queries, anytime,
                anywhere.
              </p>
            </div>

            <div className="text-center">
              <img src={icon1} alt="" className="mx-auto pb-4" />
              <div className="text-cyan-950 text-2xl font-semibold pb-4">
                Smart Guidance
              </div>
              <p className="text-cyan-950 font-semibold">
                The AI chatbot provides personalized recommendations based on
                local regulations.
              </p>
            </div>

            <div className="text-center">
              <img src={icon2} alt="" className="mx-auto pb-4" />
              <div className="text-cyan-950 text-2xl font-semibold pb-4">
                Eco-Friendly Tips
              </div>
              <p className="text-cyan-950 font-semibold ">
                Receive practical advice on reducing, reusing, and recycling
                more effectively.
              </p>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default About3;
